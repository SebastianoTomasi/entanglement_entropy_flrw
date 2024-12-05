import compute_entanglement_entropy as cee
import simulation_parameters as par

# Consider configuring PYTHONPATH instead of appending to sys.path
import sys
sys.path.append('C:/Users/sebas/custom_libraries')
import numerical_methods as nm
import input_output as io

class EntanglementEntropySimulation:
    """
    A class to simulate entanglement entropy scaling, perform linear fits, and 
    save/load simulation data.

    Attributes
    ----------
    comoving_entanglement_entropy_scaling_t : list or None
        Stores time evolution of entanglement entropy scaling once the simulation has run.
    max_errors : list or None
        Stores the maximum errors of the simulation.
    rho_for_plot_t : list or None
        Data for plotting rho over time.
    comoving_best_fits : list
        Contains the best-fit values for linear fits on the entanglement entropy data.
    comoving_fitted_slices : list
        Stores the slices of entanglement entropy data used in the linear fits.
    comoving_angular_coefficients : list
        Holds the angular coefficients (slopes) resulting from each linear fit.
    comoving_angular_coefficients_errors : list
        Stores the errors associated with each angular coefficient.
    comoving_intercepts : list
        Contains the intercepts from each linear fit.
    comoving_intercepts_errors : list
        Stores the errors associated with each intercept.
    parameters : dict
        Stores important parameters for the simulation extracted from an external
        configuration object.
    did_fit : bool
        Flag indicating whether the linear fit has been performed.
    times : list
        Time points used in the simulation, excluding the first point.

    Methods
    -------
    run()
        Executes the simulation and populates entanglement entropy scaling data.
    perform_linear_fit(suppress_report=True)
        Performs a linear fit on the simulated entanglement entropy data if available.
    load()
        Loads simulation and fit data from saved files.
    save_data()
        Saves the entanglement entropy scaling and fit data, along with simulation
        settings, to files.
    """

    def __init__(self):
        """Initialize the EntanglementEntropySimulation class."""
        self.comoving_entanglement_entropy_scaling_t = None
        self.max_errors = None
        self.rho_for_plot_t = None

        self.comoving_best_fits = []
        self.comoving_fitted_slices = []
        self.comoving_angular_coefficients = []
        self.comoving_angular_coefficients_errors = []
        self.comoving_intercepts = []
        self.comoving_intercepts_errors = []

        self.parameters = {key: getattr(par, key)
                           for key in par.important_parameters
                           if hasattr(par, key)}
        self.did_fit = False
        self.times = par.times[1:]

    def run(self):
        """Execute the simulation and populate entanglement entropy scaling data."""
        print("Simulation Settings:")
        for key, value in self.parameters.items():
            formatted_value = f"{value:.1e}" if isinstance(value, (int, float)) else value
            print(f"{key} = {formatted_value}")
        print("")

        result = cee.run()
        if len(result) == 3:
            (self.comoving_entanglement_entropy_scaling_t,
             self.max_errors,
             self.sigma_l_t_for_plot) = result
        else:
            raise ValueError("cee.run() did not return the expected number of outputs.")

    def perform_linear_fit(self, suppress_report=True):
        """
        Perform a linear fit on the simulated entanglement entropy data if available.

        Parameters
        ----------
        suppress_report : bool, optional
            If True, suppresses the report from the fitting function. Default is True.
        """
        if self.comoving_entanglement_entropy_scaling_t is None:
            print("Simulation has not been run yet. Please run the simulation before fitting.")
            return

        for entanglement_entropy_A in self.comoving_entanglement_entropy_scaling_t:
            # Perform a linear fit on the entropy data and store the results.
            (best_fit, fitted_slice, optimal_parameters, errors) = nm.fit_line(
                entanglement_entropy_A,
                par.skip_first_percent,
                par.skip_last_percent,
                suppress_report=suppress_report)

            self.comoving_best_fits.append(best_fit)
            self.comoving_fitted_slices.append(fitted_slice)
            self.comoving_angular_coefficients.append(optimal_parameters[0])
            self.comoving_angular_coefficients_errors.append(errors[0])
            self.comoving_intercepts.append(optimal_parameters[1])
            self.comoving_intercepts_errors.append(errors[1])

        self.did_fit = True

    def load(self):
        """Load and fill the class attributes from saved data files."""
        try:
            aux = io.load_data(f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t{par.fixed_name_right}")
            self.parameters = aux[0]
            data = aux[1]
            self.times = data[0]
            self.comoving_entanglement_entropy_scaling_t = data[1]
            self.rho_for_plot_t = data[2]
            self.sigma_l_t_for_plot = data[3]
        except Exception as e:
            print(f"Error loading simulation data: {e}")
            return
    
        try:
            aux = io.load_data(f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t_fit_params{par.fixed_name_right}")
            
            self.parameters = aux[0]
            for key, value in self.parameters.items():#This make sure we are using the correct parameters
                setattr(par, key, value)
            print("Simulation Settings:")
            for key, value in self.parameters.items():
                formatted_value = f"{value:.1e}" if isinstance(value, (int, float)) else value
                print(f"{key} = {formatted_value}")
            print("")
            
            data = aux[1]
            self.times = data[0]
            self.comoving_angular_coefficients = data[1]
            self.comoving_angular_coefficients_errors = data[2]
            self.comoving_intercepts = data[3]
            self.comoving_intercepts_errors = data[4]
            self.did_fit = True
        except Exception:
            print("No fit data found.")
            self.did_fit = False


    def save_data(self):
        """Save the simulation data and optionally plot settings to files."""
        io.save_data(
            data=[
                self.times,
                self.comoving_entanglement_entropy_scaling_t,
                self.rho_for_plot_t,
                self.sigma_l_t_for_plot
            ],
            path=f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t{par.fixed_name_right}",
            header=self.parameters
        )
    
        if self.did_fit:
            io.save_data(
                data=[
                    self.times,
                    self.comoving_angular_coefficients,
                    self.comoving_angular_coefficients_errors,
                    self.comoving_intercepts,
                    self.comoving_intercepts_errors
                ],
                path=f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t_fit_params{par.fixed_name_right}",
                header=self.parameters
            )
    
        if par.save_plots:
            file_path = f"{par.fixed_name_left}simulation_settings{par.fixed_name_right}.txt"
            with open(file_path, 'w') as file:
                file.write("Simulation Settings:\n")
                for key, value in self.parameters.items():
                    formatted_value = f"{value:.3e}" if isinstance(value, (int, float)) else value
                    file.write(f"{key} = {formatted_value}\n")

