import compute_entanglement_entropy as cee
import simulation_parameters as par

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

    comoving_best_fits : list
        Contains the best-fit values for linear fits on the entanglement entropy data.

    comoving_fitted_slices : list
        Stores the slices of entanglement entropy data used in the linear fits.

    comoving_angular_coefficients : list
        Holds the angular coefficients (slopes) resulting from each linear fit.

    comoving_angular_coefficients_errors : list
        Stores the errors associated with each angular coefficient.

    comoving_intercepts : list
        Contains the comoving_intercepts from each linear fit.

    comoving_intercepts_errors : list
        Stores the errors associated with each intercept.

    parameters : dict
        Stores important parameters for the simulation extracted from an external
        configuration object.

    Methods
    -------
    run()
        Executes the simulation and populates entanglement entropy scaling data.

    perform_linear_fit()
        Performs a linear fit on the simulated entanglement entropy data if available.

    load()
        Loads simulation and fit data from saved files.

    save_data()
        Saves the entanglement entropy scaling and fit data, along with simulation
        settings, to files.
    """
    def __init__(self):
        
        self.comoving_entanglement_entropy_scaling_t = None

        self.max_errors=None
        self.rho_for_plot_t=None
        
        self.comoving_best_fits = []
        self.comoving_fitted_slices = []
        self.comoving_angular_coefficients = []
        self.comoving_angular_coefficients_errors = []
        self.comoving_intercepts = []
        self.comoving_intercepts_errors = []

        
        self.parameters = {key: getattr(par, key)
                           for key in par.important_parameters
                           if hasattr(par, key) }
        self.did_fit=False
        self.times=par.times[1:]
        
    def run(self):
        
        print("Simulation Settings:")
        for key, value in self.parameters.items():
            formatted_value = f"{value:.1e}" if isinstance(value, (int, float)) else value
            print(f"{key} = {formatted_value}")
        print("")
            
        self.comoving_entanglement_entropy_scaling_t, self.max_errors, self.sigma_l_t_for_plot =cee.run()
        
    def perform_linear_fit(self,suppress_report=True):
        """Can only run this if the simulation has been run."""
        if self.comoving_entanglement_entropy_scaling_t is not None:
            for entanglement_entropy_A in self.comoving_entanglement_entropy_scaling_t:
                """Perform a linear fit on the entropy data and store the results."""
                comoving_best_fit, comoving_fitted_slice, comoving_optimal_parameters, comoving_error = nm.fit_line(
                    entanglement_entropy_A,
                    par.skip_first_percent, par.skip_last_percent,
                    suppress_report=suppress_report)
                
                self.comoving_best_fits.append(comoving_best_fit)
                self.comoving_fitted_slices.append(comoving_fitted_slice)
                self.comoving_angular_coefficients.append(comoving_optimal_parameters[0])
                self.comoving_angular_coefficients_errors.append(comoving_error[0])
                self.comoving_intercepts.append(comoving_optimal_parameters[1])
                self.comoving_intercepts_errors.append(comoving_error[1])
            self.did_fit=True

    def load(self):
        """Load and fill the class attributes from saved data files."""
        aux = io.load_data(f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t_"+par.cosmology)
        print(aux)
        self.parameters = aux[0]
        self.times=aux[1][0]
        self.comoving_entanglement_entropy_scaling_t=aux[1][1]
        self.rho_for_plot_t=aux[2]
        
        aux = io.load_data(f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t_fit_params_"+par.cosmology)
        try:
            self.comoving_angular_coefficients = aux[1][1]
            self.comoving_angular_coefficients_errors = aux[1][2]
            self.did_fit=True
        except:
            print("No fit found.")
    
    
    
    def save_data(self):
        """Save the simulation data and optionally plot settings to files."""
        io.save_data(data=[self.times,self.comoving_entanglement_entropy_scaling_t,self.rho_for_plot_t], 
                     path=f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t_"+par.cosmology,
                     header=self.parameters)
        if self.did_fit:
            io.save_data(data=[self.times,
                               self.comoving_angular_coefficients,
                               self.comoving_angular_coefficients_errors,
                               self.comoving_intercepts,
                               self.comoving_intercepts_errors],
                         path=f"{par.fixed_name_left}comoving_entanglement_entropy_scaling_t_fit_params_"+par.cosmology,
                         header=self.parameters)
    
        if par.save_plots:
            file_path=f"{par.fixed_name_left}simulation_settings_"+par.cosmology+".txt"
            with open(file_path, 'w') as file:
                file.write("Simulation Settings:\n")
                for key, value in self.parameters.items():
                    formatted_value = f"{value:.3e}" if isinstance(value, (int, float)) else value
                    file.write(f"{key} = {formatted_value}\n")

