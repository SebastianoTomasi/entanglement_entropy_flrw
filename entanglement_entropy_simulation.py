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
    entanglement_entropy_scaling_t : list or None
        Stores time evolution of entanglement entropy scaling once the simulation has run.

    best_fits : list
        Contains the best-fit values for linear fits on the entanglement entropy data.

    fitted_slices : list
        Stores the slices of entanglement entropy data used in the linear fits.

    angular_coefficients : list
        Holds the angular coefficients (slopes) resulting from each linear fit.

    angular_coefficients_errors : list
        Stores the errors associated with each angular coefficient.

    intercepts : list
        Contains the intercepts from each linear fit.

    intercepts_errors : list
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
        
        self.entanglement_entropy_scaling_t = None
        self.max_errors=None
        
        self.best_fits = []
        self.fitted_slices = []
        self.angular_coefficients = []
        self.angular_coefficients_errors = []
        self.intercepts = []
        self.intercepts_errors = []
        
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
            
        self.entanglement_entropy_scaling_t, self.max_errors =cee.run()
        
    def perform_linear_fit(self,suppress_report=True):
        """Can only run this if the simulation has been run."""
        if self.entanglement_entropy_scaling_t is not None:
            for entanglement_entropy_A in self.entanglement_entropy_scaling_t:
                """Perform a linear fit on the entropy data and store the results."""
                best_fit, fitted_slice, optimal_parameters, error = nm.fit_line(
                    entanglement_entropy_A,
                    par.skip_first_percent, par.skip_last_percent,
                    suppress_report=suppress_report)
                
                self.best_fits.append(best_fit)
                self.fitted_slices.append(fitted_slice)
                self.angular_coefficients.append(optimal_parameters[0])
                self.angular_coefficients_errors.append(error[0])
                self.intercepts.append(optimal_parameters[1])
                self.intercepts_errors.append(error[1])
            self.did_fit=True

    def load(self):
        """Load and fill the class attributes from saved data files."""
        aux = io.load_data(par.save_data_dir + "/entanglement_entropy_scaling_t_"+par.cosmology)
        print(aux)
        self.parameters = aux[0]
        self.times=aux[1][0]
        self.entanglement_entropy_scaling_t=aux[1][1]
        
        aux = io.load_data(par.save_data_dir + "/entanglement_entropy_scaling_t_fit_params_"+par.cosmology)
        self.angular_coefficients = aux[1][1]
        self.angular_coefficients_errors = aux[1][2]
        self.did_fit=True
        # except:
        #     print("No fit found.")
    
    
    
    def save_data(self):
        """Save the simulation data and optionally plot settings to files."""
        io.save_data(data=[self.times,self.entanglement_entropy_scaling_t], 
                     path=par.save_data_dir + "/entanglement_entropy_scaling_t_"+par.cosmology,
                     header=self.parameters)
        if self.did_fit:
            io.save_data(data=[self.times,
                               self.angular_coefficients,
                               self.angular_coefficients_errors,
                               self.intercepts,
                               self.intercepts_errors],
                         path=par.save_data_dir + "/entanglement_entropy_scaling_t_fit_params_"+par.cosmology,
                         header=self.parameters)
    
        if par.save_plots:
            file_path=par.save_plot_dir + "/simulation_settings_"+par.cosmology+".txt"
            with open(file_path, 'w') as file:
                file.write("Simulation Settings:\n")
                for key, value in self.parameters.items():
                    formatted_value = f"{value:.3e}" if isinstance(value, (int, float)) else value
                    file.write(f"{key} = {formatted_value}\n")

