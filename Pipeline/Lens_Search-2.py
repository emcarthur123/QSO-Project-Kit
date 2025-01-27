import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
from scipy.optimize import Bounds
import matplotlib as mpl
import warnings

# Suppress all RuntimeWarnings
warnings.simplefilter("ignore", category=RuntimeWarning)

class QSOFluxProcessor:
    classifier_model = None
    redshift_model = None
    waves = None

    @classmethod
    def setup(cls):
            """
    Class-level setup method to initialize models and data only once.
    This method ensures that the classifier model, redshift model, 
    and wave data are loaded only at the class level, avoiding redundant loading.

    Attributes:
        cls.classifier_model: The machine learning model used for classification.
        cls.redshift_model: The machine learning model used for redshift prediction.
        cls.waves: Pre-loaded wave data from a NumPy file.
    """
        if cls.classifier_model is None:
            cls.classifier_model = cls.build_classifier_model()
        if cls.redshift_model is None:
            cls.redshift_model = cls.build_redshift_model()
        if cls.waves is None:
            cls.waves = np.load('wave.npy', allow_pickle=True)

    @staticmethod
    def build_classifier_model(input_layer=50, n_node_CNN=[50, 50, 100, 100, 100], n_node_FC=[30, 25]):
        """ Builds the model """
        input_shape = (7781, 1)
        inputs = Input(shape=input_shape)
        x = Conv1D(input_layer, kernel_size=5, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        for k in n_node_CNN:
            x = Conv1D(k, kernel_size=5, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        for n in n_node_FC:
            x = Dense(n, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights('Weights/Phase2.h5')
        return model

    @staticmethod
    def build_redshift_model(input_layer=50, n_node_CNN=[50, 50, 100, 100, 100], n_node_FC=[30, 25]):
        """ Builds the model """
        input_shape = (7781, 1)
        inputs = Input(shape=input_shape)
        x = Conv1D(input_layer, kernel_size=5, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        for k in n_node_CNN:
            x = Conv1D(k, kernel_size=5, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        for n in n_node_FC:
            x = Dense(n, activation='relu')(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights('Weights/Redshift.h5')
        return model

    @staticmethod
    def preprocess_data(array):
        """ Staandardizes each input spectrum """
        standardized = np.array([((flux - np.mean(flux)) / np.std(flux)) for flux in array])
        stn = np.array([x / np.max(np.abs(x)) for x in standardized])
        stn = np.expand_dims(stn, axis=-1)
        return stn

    @classmethod
    def classify(cls, fluxes):
        cls.setup()
        preprocessed = cls.preprocess_data(fluxes)
        return cls.classifier_model.predict(preprocessed)

    @classmethod
    def predict_redshift(cls, fluxes):
        cls.setup()
        preprocessed = cls.preprocess_data(fluxes)
        return cls.redshift_model.predict(preprocessed)
    
    @staticmethod
    def gaussian(x, amplitude, mean, sigma):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def model(x_value, amp1, mean1, amp2, mean2, sigma, cont):
        gaussian1 = QSOFluxProcessor.gaussian(x_value, amp1, mean1, sigma)
        gaussian2 = QSOFluxProcessor.gaussian(x_value, amp2, mean2, sigma)
        return gaussian1 + gaussian2 + cont
    
    @staticmethod
    def residuals(params, x, y):
        return y - QSOFluxProcessor.model(x, *params)

    @classmethod
    def fit_gaussian(cls, flux, wave, redshift):
        """
        Fits a Gaussian model to the OII emission lines in a given spectrum.

        This function identifies the OII emission region based on the input redshift, 
        truncates the spectrum to focus on that region, smooths the flux, and fits a 
        double-Gaussian model to the emission lines using least-squares optimization. 
        It calculates the fitted redshift, parameter uncertainties, and other diagnostics.

        Args:
            flux (numpy.ndarray): Flux values of the spectrum.
            wave (numpy.ndarray): Wavelength values corresponding to the flux.
            redshift (float): Initial estimate of the redshift.

        Returns:
            tuple: Contains fitted redshift, optimized parameters, parameter errors, 
            redshift error, fitting cost, fitting status flag, truncated wavelength, 
            truncated flux, mask, and smoothed flux.
        """
        try:
            # Define redshift range and OII wavelengths
            redshift_low = redshift - 0.1
            redshift_high = redshift + 0.1
            oii1_wavelength = 3726
            oii2_wavelength = 3729

            # Truncate the wavelength range based on redshift
            wavelength_low = oii1_wavelength * (1 + redshift_low)
            wavelength_high = oii2_wavelength * (1 + redshift_high)
            mask = (wave >= wavelength_low) & (wave <= wavelength_high)
            truncated_wave = wave[mask]
            truncated_flux = flux[mask]

            # Check for empty arrays
            if len(truncated_wave) == 0 or len(truncated_flux) == 0:
                raise ValueError("Truncated arrays are empty, cannot fit.")

            # Smooth flux
            smoothed_flux = gaussian_filter1d(truncated_flux, sigma=1)

            # Initial parameters
            initial_mean1 = truncated_wave[np.argmax(smoothed_flux)]
            initial_mean2 = initial_mean1 + 3  # Ensures separation between means
            initial_params = [
                np.max(smoothed_flux) / 2, initial_mean1,
                np.max(smoothed_flux) / 3, initial_mean2,
                2, np.median(smoothed_flux)
            ]

            # Bound adjustments to avoid overlap and ensure proper ordering
            separation_factor = 0.1  # Small factor to ensure bounds don't overlap
            bounds = Bounds(
                [0, initial_mean1 - 5 - separation_factor, 0, initial_mean2 - 2 - separation_factor, 0.5, np.min(smoothed_flux)],
                [np.max(smoothed_flux), initial_mean1 + 5 + separation_factor, np.max(smoothed_flux), initial_mean2 + 5 + separation_factor, 5, np.max(smoothed_flux)]
            )

            # Perform least-squares fitting
            result = least_squares(
                cls.residuals, initial_params, args=(truncated_wave, smoothed_flux), bounds=bounds
            )

            # Extract fitted parameters and calculate covariance
            opt_params = result.x
            jacobian = result.jac
            covariance_matrix = np.linalg.pinv(jacobian.T @ jacobian)  # Use pseudoinverse to avoid singularity
            param_errors = np.sqrt(np.diag(covariance_matrix))
            # Calculate fitted redshift and error
            fitted_redshift = (opt_params[1] - oii1_wavelength) / oii1_wavelength
            redshift_error = param_errors[1] / oii1_wavelength

            return fitted_redshift, opt_params, param_errors, redshift_error, result.cost, 0, truncated_wave, truncated_flux, mask, smoothed_flux

        except Exception as e:
            return np.nan, [np.nan] * 6, [np.nan] * 6, np.inf, 1, 1, truncated_wave, truncated_flux, mask, smoothed_flux

    @staticmethod   
    def read_data(filename):
        """Read the data from the provided text file and return the relevant columns."""
        wavelengths = []
        eq_widths = []

        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, start=41):  
                parts = line.split()
                if len(parts) >= 12:  
                    try:
                        observed_wavelength = float(parts[0].strip())
                        equivalent_width = float(parts[6].strip())
                        wavelengths.append(observed_wavelength)  
                        eq_widths.append(equivalent_width)  
                    except ValueError:
                        pass

        return np.array(wavelengths), np.array(eq_widths)
    
    @classmethod   
    def plot_gaussian_fit_with_subplots(cls, flux, wave, qso_z, fitted_redshift, targetid, snr, save_path=None, plot_qso_lines=True, show_plot=True):
        
        """
        Plot a Gaussian fit with subplots for visualizing spectral data and fitted models.

        Parameters:
        - cls: The class containing the Gaussian fitting and QSO processing methods.
        - flux: Observed flux values for the spectrum.
        - wave: Corresponding wavelength values for the spectrum.
        - qso_z: Redshift of the QSO.
        - fitted_redshift: Redshift determined by the fitting process.
        - targetid: Identifier for the target object.
        - snr: Signal-to-noise ratio of the data.
        - save_path: Optional path to save the plot as a file.
        - plot_qso_lines: Flag to control whether to overlay QSO emission lines.
        - show_plot: Flag to control whether to display the plot.

        Returns:
        - None
        """
        # Set Matplotlib configurations for styling
        mpl.rcParams["font.family"] = "Serif"
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["axes.linewidth"] = 2
        mpl.rcParams["xtick.major.size"] = 5
        mpl.rcParams["xtick.major.width"] = 1
        mpl.rcParams["ytick.major.size"] = 5
        mpl.rcParams["ytick.major.width"] = 1
        mpl.rcParams["legend.fontsize"] = 10
        mpl.rcParams["xtick.direction"] = "in"
        mpl.rcParams["ytick.direction"] = "in"
        mpl.rcParams["ytick.right"] = True
        mpl.rcParams["xtick.top"] = True

        filename = 'SDSS2.txt'  # Replace with your actual filename
        #wave = np.load('wave.npy', allow_pickle=True)
        # Read data
        wavelengths, eq_widths = QSOFluxProcessor.read_data(filename)

        fitted_redshift, opt_params, param_errors, redshift_error, best_chi_squared, flag, truncated_wave, truncated_flux, mask, smoothed_flux = cls.fit_gaussian(flux, wave, fitted_redshift)

        # Unpack the optimized parameters
        amp1, mean1, amp2, mean2, sigma, cont = opt_params

        # Generate the fitted model for the zoom-in plot
        fitted_flux = QSOFluxProcessor.model(truncated_wave, amp1, mean1, amp2, mean2, sigma, cont)
        gaussian1 = QSOFluxProcessor.gaussian(truncated_wave, amp1, mean1, sigma)
        gaussian2 = QSOFluxProcessor.gaussian(truncated_wave, amp2, mean2, sigma)

        # Create a gridspec layout: 2 rows, 3 columns (top span & bottom split)
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.5], hspace=0.3, wspace=0.4)

        # Full spectrum (Top - spans all columns)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(wave, flux, 'k-', linewidth=0.8, label='Observed Flux')

        # Check if each emission line is within the wave limits before plotting
        lines = [
            (3727 * (1 + fitted_redshift), 'OII Doublet', 'r', '--'),
            (4861.363 * (1 + fitted_redshift), 'H-beta', 'r', '-'),
            (5008.239 * (1 + fitted_redshift), '[OIII]', 'r', '-.'),
            (4960.295 * (1 + fitted_redshift), None, 'r', '-.')
        ]

        for line in lines:
            line_pos, label, color, linestyle = line
            if wave.min() <= line_pos <= wave.max():
                ax1.axvline(x=line_pos, color=color, linestyle=linestyle, label=label)

        if plot_qso_lines:  # Only plot if the flag is set
            # Plot emission lines as shaded regions
            label_added = False
            for i in range(len(wavelengths)):
                emission_line = wavelengths[i] * (1 + qso_z)
                if wave.min() <= emission_line <= wave.max():
                    width = eq_widths[i]  
                    ax1.fill_betweenx(
                        y=[flux.min(), flux.max()],  
                        x1=emission_line - width / 2,
                        x2=emission_line + width / 2,
                        color='black',
                        alpha=0.3,
                        label='QSO Emission Line' if not label_added else ""
                    )
                    label_added = True

        ax1.set_ylim(flux.min(), flux.max())
        ax1.set_xlabel('Wavelength [Å]')
        ax1.set_ylabel("$Flux [ 10^{-17} {erg s}^{-1} {cm}^{-2} Å^{-1}]$")
        ax1.set_title(f'TargetID: {targetid}, Z_qso: {qso_z:.4f}, Z_elg: {fitted_redshift:.4f}, SNR: {snr:.4f}')
        ax1.legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=True)

        # Zoom-in on OII doublet region (±100 around 3727 Å)
        oii_center = 3727 * (1 + fitted_redshift)
        oii_mask = (truncated_wave >= oii_center - 100) & (truncated_wave <= oii_center + 100)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(truncated_wave[oii_mask], truncated_flux[oii_mask], 'k-', label='Observed Flux')
        ax2.plot(truncated_wave[oii_mask], fitted_flux[oii_mask], 'cyan', label='Full Model')
        ax2.plot(truncated_wave[oii_mask], gaussian1[oii_mask], 'b--', label='Gaussian 1')
        ax2.plot(truncated_wave[oii_mask], gaussian2[oii_mask], 'g--', label='Gaussian 2')

        # Check for OII line in truncated wave range
        if wave.min() <= 3727 * (1 + fitted_redshift) <= wave.max():
            ax2.axvline(x=3727 * (1 + fitted_redshift), color='r', linestyle='--', label='OII Doublet')

        label_added2 = False
        if plot_qso_lines:  # Only plot if the flag is set
            for i in range(len(wavelengths)):
                if wavelengths[i] * (1 + qso_z) >= truncated_wave[oii_mask].min() and wavelengths[i] * (1 + qso_z) <= truncated_wave[oii_mask].max():
                    center = wavelengths[i] * (1 + qso_z)
                    width = eq_widths[i]
                    ax2.fill_betweenx(
                        y=[truncated_flux.min(), truncated_flux.max()],
                        x1=center - width / 2,
                        x2=center + width / 2,
                        color='black',
                        alpha=0.3,
                        label='QSO Emission Line' if not label_added2 else ""
                    )
                    label_added2 = True

        ax2.set_ylim(truncated_flux[oii_mask].min(), truncated_flux[oii_mask].max())
        ax2.set_xlabel('Wavelength [Å]')
        ax2.set_ylabel("$Flux [ 10^{-17} { erg s}^{-1} {cm}^{-2} Å^{-1}]$")
        ax2.set_title('OII Doublet')
        ax2.legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=True)

        # OIII and H-beta lines (Bottom-Right, spans 2 columns)
        oiii_hbeta_mask = (wave >= 4850 * (1 + fitted_redshift)) & (wave <= 5010 * (1 + fitted_redshift))
        oiii_hbeta_wave = wave[oiii_hbeta_mask]
        oiii_hbeta_flux = flux[oiii_hbeta_mask]

        ax3 = fig.add_subplot(gs[1, 1:])
        ax3.plot(oiii_hbeta_wave, oiii_hbeta_flux, 'k-', label='OIII & H-beta Region')
        label_added3 = False

        # Ensure that oiii_hbeta_wave is not empty before accessing min and max
        if oiii_hbeta_wave.size > 0:
            if 4861.363 * (1 + fitted_redshift) >= oiii_hbeta_wave.min() and 4861.363 * (1 + fitted_redshift) <= oiii_hbeta_wave.max():
                ax3.axvline(x=4861.363 * (1 + fitted_redshift), color='r', linestyle='-', label='H-beta')
            if 5008.239 * (1 + fitted_redshift) >= oiii_hbeta_wave.min() and 5008.239 * (1 + fitted_redshift) <= oiii_hbeta_wave.max():
                ax3.axvline(x=5008.239 * (1 + fitted_redshift), color='r', linestyle='-.', label='[OIII]')
            if 4960.295 * (1 + fitted_redshift) >= oiii_hbeta_wave.min() and 4960.295 * (1 + fitted_redshift) <= oiii_hbeta_wave.max():
                ax3.axvline(x=4960.295 * (1 + fitted_redshift), color='r', linestyle='-.')

            if plot_qso_lines:  # Only plot if the flag is set
                for i in range(len(wavelengths)):
                    emission_line = wavelengths[i] * (1 + qso_z)
                    if oiii_hbeta_wave.min() <= emission_line <= oiii_hbeta_wave.max():
                        width = eq_widths[i]
                        ax3.fill_betweenx(
                            y=[oiii_hbeta_flux.min(), oiii_hbeta_flux.max()],
                            x1=emission_line - width / 2,
                            x2=emission_line + width / 2,
                            color='black',
                            alpha=0.3,
                            label='QSO Emission Line' if not label_added3 else ""
                        )
                        label_added3 = True

            ax3.set_ylim(oiii_hbeta_flux.min(), oiii_hbeta_flux.max())
        else:
            ax3.set_ylim(0, 1)  # or set to desired default when no data is present

        ax3.set_xlabel('Wavelength [Å]')
        ax3.set_ylabel("$Flux [ 10^{-17} { erg s}^{-1} {cm}^{-2} Å^{-1}]$")
        ax3.set_title('H-beta & OIII Region')
        ax3.legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        if show_plot:
            plt.show()
        plt.close(fig)


    @classmethod
    def process(cls, fluxes, names=None, noise=None, z_qsos=None):
        """
        Processes a batch of flux data to classify objects as 'Lens' or 'QSO', predict their redshifts, 
        fit Gaussian models to their spectra, and compute additional metrics such as SNR and redshift errors.

        Parameters:
            fluxes (list or np.ndarray): Array of flux data for the objects to process.
            names (list, optional): List of names or identifiers for the objects.
            noise (np.ndarray, optional): Noise values associated with the flux data for SNR calculation.
            z_qsos (list or np.ndarray, optional): True redshifts of QSOs, if available.

        Returns:
            pd.DataFrame: A DataFrame containing classification, redshift predictions, 
                          Gaussian fit parameters, and additional metrics for each object.
        """
        cls.setup()
        predictions_class = cls.classify(fluxes)
        predictions_z = cls.predict_redshift(fluxes)

        results = []
        for i, flux in enumerate(fluxes):
            wave = cls.waves
            predicted_redshift = predictions_z[i][0]
            z_qso = z_qsos[i] if z_qsos is not None else np.nan

            fitted_redshift, opt_params, param_errors, redshift_error, best_chi_squared, flag, truncated_wave, truncated_flux, mask, smoothed_flux = cls.fit_gaussian(flux, wave, predicted_redshift)

            score = predictions_class[i][0]
            classification = 'Lens' if score > 0.5 else 'QSO'

            name = names[i] if names is not None else f"QSO_{i+1}"

            SNR = np.nan
            if noise is not None and flag == 0:
                if np.any(noise[i] > 0):  # Check if any value in the noise array is greater than zero
                    current_noise = 1 / np.sqrt(noise[i])  # Adjust this to the appropriate index
                else:
                    current_noise = 0  # or some default value
                fitted_mean1 = opt_params[1]
                idx_fitted_mean1 = np.abs(truncated_wave - fitted_mean1).argmin()
                low_index = max(idx_fitted_mean1 - 5, 0)
                high_index = min(idx_fitted_mean1 + 5, len(truncated_wave))

                narrow_truncated_flux = truncated_flux[low_index:high_index]
                narrow_truncated_noise = current_noise[mask][low_index:high_index]
                if np.sum(narrow_truncated_noise[np.isfinite(narrow_truncated_noise)]) > 0:
                    SNR = np.sum(narrow_truncated_flux) / np.sum(narrow_truncated_noise[np.isfinite(narrow_truncated_noise)])

            if flag == 0:
                amp1, mean1, sigma1, amp2, mean2, cont = opt_params
                err_amp1, err_mean1, err_sigma1, err_amp2, err_mean2, err_cont = param_errors
                results.append([name, z_qso, score, classification, predicted_redshift, fitted_redshift, redshift_error, best_chi_squared, amp1, err_amp1, mean1, err_mean1, sigma1, err_sigma1, amp2, err_amp2, mean2, err_mean2, cont, err_cont, SNR])
            else:
                results.append([name, z_qso, score, classification, predicted_redshift, np.nan, np.inf, np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, SNR])

        columns = ['Name', 'z_qso', 'Score', 'Classification', 'Predicted Redshift', 'Fitted Redshift', 'Redshift Error', 'Best Chi-Squared', 'Amp1', 'Err_Amp1', 'Mean1', 'Err_Mean1', 'Sigma1', 'Err_Sigma1', 'Amp2', 'Err_Amp2', 'Mean2', 'Err_Mean2', 'Cont', 'Err_Cont', 'SNR']

        return pd.DataFrame(results, columns=columns)

