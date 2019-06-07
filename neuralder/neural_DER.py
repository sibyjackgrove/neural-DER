"""Generate training samples from PV-DER simulation."""

import random
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,LSTM

#Simulation modules
from pvder.DER_components_single_phase import SolarPV_DER_SinglePhase
from pvder.DER_components_three_phase  import SolarPV_DER_ThreePhase
from pvder.grid_components import Grid
from pvder.dynamic_simulation import DynamicSimulation
from pvder.simulation_events import SimulationEvents


class NeuralDERData():
    """
    Class generating training data for PV-DER.
    
    Attributes:
         DER_count (int): Number of `SolarPV_DER_SinglePhase` instances.
         Ioverload (float): Overload current rating of inverter.
    """
    

    def __init__(self,SINGLE_PHASE = False,rating = 50.0e3,SEQUENCE=False,                 
                 verbosity='INFO'):
        """Creates an instance of `SolarPV_DER`.
        
        Args:
          SINGLE_PHASE (bool): Specify whether single phase DER instance should be created.
          rating (float): A scalar specifying the rated power (VA) of the DER.          
          SEQUENCE (bool): Specify whether data should be sequential (for use with RNN's).
          t_event_start (float): Time (s) at which simulation events end.
          t_event_end (float): Time (s) at which simulation events end.
          tStop (float): Time (s) at which simulation ends.
          
        Raises:
          AssertError: If SINGLE_PHASE or SEQUENCE parameter are not boolean.
        """
        
        assert type(SINGLE_PHASE) is bool, 'SINGLE_PHASE should be boolean!'
        assert type(SEQUENCE) is bool, 'SEQUENCE should be boolean!'
        
        self._SINGLE_PHASE = SINGLE_PHASE
        self._rating = rating
        self._SEQUENCE = SEQUENCE
        self.verbosity = verbosity
        
    def generate_training_data(self,n_instances = 3,t_event_start=2,t_event_end=9,tStop=10,):
        """Create training data from specified number of instances."""
                 
        self.create_DER_simulation_instances(n_instances)
        self.generate_events(t_event_start,t_event_end)
        self.run_sim_instances(tStop)
                 
        return self.extract_data()

    def create_DER_simulation_instances(self,n_instances):
        """Create specified number of DER instances."""
                 
        assert n_instances >= 1, 'Number of simulation instances should be greater than or equal to one!'
                 
        self.events_list = []
        self.grid_list =[]
        self.DER_list =[]
        self.sim_list =[]
                 
        print('{} DER model instances will be created.'.format(n_instances))
                 
        for instance in range(n_instances):
            print('Creating Instance:',instance+1)
                 
            self.events_list.append(SimulationEvents())
            self.grid_list.append(Grid(events=self.events_list[instance],unbalance_ratio_b=1.0,unbalance_ratio_c=1.0))
        
            if self._SINGLE_PHASE:
                self.DER_list.append(SolarPV_DER_SinglePhase(grid_model = self.grid_list[instance],events = self.events_list[instance],
                                                             Sinverter_rated = self._rating,standAlone = True,
                                                             STEADY_STATE_INITIALIZATION=True,verbosity='INFO'))
                
            else:
                self.DER_list.append(SolarPV_DER_ThreePhase(grid_model=self.grid_list[instance],events=self.events_list[instance],
                                                            Sinverter_rated = self._rating,standAlone = True,
                                                            STEADY_STATE_INITIALIZATION=True,verbosity='INFO'))
            
            self.sim_list.append(DynamicSimulation(grid_model=self.grid_list[instance],PV_model=self.DER_list[instance],events = self.events_list[instance]))

    def generate_events(self,t_event_start,t_event_end,t_event_step=1.0):
        """Create random events."""
        
        for event_instance in self.events_list:
            event_instance.create_random_events(t_event_start,t_event_end,t_event_step,events_type=['insolation','voltage'])
        """
        for event_instance in events_list:
            for time in range(t_event_start,t_event_end+1):
                if random.random() > 0.5:  #Add either solar or grid event
                   Sinsol_random = random.randint(50, 100)
                   event_instance.add_solar_event(time,Sinsol_random,298.15)
                   event_instance.create_random_insolation_events()
                else:
                   Vg_random = round(random.uniform(0.95, 1.02),2)
                   event_instance.add_grid_event(time,Vg_random,60.0)
        """
        if self.verbosity == 'DEBUG':
            for event_instance in self.events_list:
                event_instance.show_events()

    def run_sim_instances(self,tStop=10.0):
        """Run all simulation instances."""
    
        sim_failed_list = []
    
        for DER_instance in self.DER_list:
            DER_instance.VOLT_VAR_ENABLE = False
            DER_instance.LVRT_ENABLE = False
            DER_instance.LFRT_ENABLE = False
            DER_instance.DO_EXTRA_CALCULATIONS = True
            DER_instance.MPPT_ENABLE=False
            DER_instance.RAMP_ENABLE = False
        
        for sim_instance in self.sim_list:
            sim_instance.jacFlag = True
            sim_instance.DEBUG_SIMULATION = False
            sim_instance.DEBUG_VOLTAGES = True
            sim_instance.DEBUG_CURRENTS = True
            sim_instance.DEBUG_POWER = False
            sim_instance.DEBUG_CONTROLLERS  = True
            sim_instance.DEBUG_PLL = False
            sim_instance.PER_UNIT = True
            sim_instance.DEBUG_SOLVER  = False
            sim_instance.tStop = tStop
            sim_instance.tInc = 0.001

        for sim_instance in self.sim_list:
            try:
               sim_instance.run_simulation()
            
            except ValueError:# ValueError:
               print('Adding {} to failed list due to simulation error!'.format(sim_instance.name))
               sim_failed_list.append(sim_instance)
    
        print('{} simulation instances failed!'.format(len(sim_failed_list)))
    
        for sim_instance in sim_failed_list:
            print('Removing failed instance {} from simulation list!'.format(sim_instance.name))
            self.sim_list.remove(sim_instance)
    
    def extract_data(self):
        """Create training dataset."""
    
        Vdc_t = []
        iaR_t = []
        iaI_t = []
        maR_t = []
        maI_t = []
        xaR_t = []
        xaI_t = []
        uaR_t = []
        uaI_t = []
        xDC_t = []
        xQ_t = []
        vagR_t = []
        vagI_t = []
        Sinsol_t =  []
    
        for sim in self.sim_list:
            print("Extracting data from  {}".format(sim.name))

            Vdc_t.append(sim.Vdc_t)
            iaR_t.append(sim.iaR_t)
            iaI_t.append(sim.iaI_t)
            maR_t.append(sim.maR_t)
            maI_t.append(sim.maI_t)
            xaR_t.append(sim.xaR_t)
            xaI_t.append(sim.xaI_t)
            uaR_t.append(sim.uaR_t)
            uaI_t.append(sim.uaI_t)
            xDC_t.append(sim.xDC_t)
            xQ_t.append(sim.xQ_t)
            vagR_t.append(sim.vagR_t)
            vagI_t.append(sim.vagI_t)
            Sinsol_t.append(sim.Sinsol_t)

            #print(sim.Vdc_t.shape)
            #print(sim.iaR_t.shape)

        #print(Vdc_t.shape)   
        Vdc_t = np.concatenate(Vdc_t)
        iaR_t = np.concatenate(iaR_t)
        iaI_t = np.concatenate(iaI_t)
        maR_t = np.concatenate(maR_t)
        maI_t = np.concatenate(maI_t)
        xaR_t = np.concatenate(xaR_t)
        xaI_t = np.concatenate(xaI_t)
        uaR_t = np.concatenate(uaR_t)
        uaI_t = np.concatenate(uaI_t)
        xDC_t = np.concatenate(xDC_t)
        xQ_t = np.concatenate(xQ_t)
        vagR_t = np.concatenate(vagR_t)
        vagI_t = np.concatenate(vagI_t)
        Sinsol_t = np.concatenate(Sinsol_t)

        self.Xtrain_raw = np.stack((Vdc_t,
                       iaR_t,iaI_t,
                       xaR_t,xaI_t,
                       uaR_t,uaI_t,
                       xDC_t,xQ_t,
                       vagR_t,vagI_t,Sinsol_t),axis =-1) #maR_t,maI_t,
    
        self.Ytrain_raw = np.stack((Vdc_t,
                       iaR_t,iaI_t,
                       xaR_t,xaI_t,
                       uaR_t,uaI_t,
                       xDC_t,xQ_t),axis =-1) #maR_t,maI_t,
                       
        print('Shape after concat:{},{}'.format(self.Xtrain_raw.shape,self.Ytrain_raw.shape))
        
        self.Xtrain = self.Xtrain_raw[0:-2,:]
        self.Ytrain = self.Ytrain_raw[1:-1,:]
    
        assert np.array_equal(self.Ytrain[0,:], self.Xtrain[1,0:9]), 'The first vector in Ytrain should be equal to second vector in Xtrain!'
        assert len(self.Xtrain) == len(self.Ytrain), 'Number of input and target samples should be equal!'
    
        if self._SEQUENCE:
            self.Xtrain = self.Xtrain.reshape(-1,2,12)
            self.Ytrain = self.Ytrain.reshape(-1,2,9)        
    
        print('Input data shape:{},Target data shape:{}'.format(self.Xtrain.shape,self.Ytrain.shape))
    
        return self.Xtrain.astype('float32'),self.Ytrain.astype('float32')    

    

class NeuralDERModel():
    """
    Class generating training data for PV-DER.
    
    Attributes:
         _DERmodel_spec: Default specifications for DER NN model.
    """
    
    _DERmodel_spec = {'dense':{'default_units':50,'num_layers':1,'num_units':[50],'ADD_BATCHNORM':False},
                     'LSTM':{'default_units':20,'num_layers':1,'num_units':[20]}}  #Time delay between events

    def __init__(self,DERdata,num_layers=None,ADD_BATCHNORM=None,num_units=None,default_units=None):
        """Creates an instance of `SolarPV_DER`.
        
        Args:
          DERdata: An instance of `NeuralDERData`.
          rating (float): A scalar specifying the rated power (VA) of the DER.          
          
        Raises:
          AssertError: If SINGLE_PHASE or SEQUENCE parameter are not boolean.
        """
        
        self.DERdata = DERdata
        
        self.num_features = self.DERdata.Xtrain.shape[-1]
        self.num_outputs = self.DERdata.Ytrain.shape[-1]
        
        if len(self.DERdata.Xtrain.shape) == 2:
            self.model_type = 'dense'
        elif len(self.DERdata.Xtrain.shape) == 3:
            self.sequence_length =  self.DERdata.Xtrain.shape[1]
            self.model_type = 'LSTM'        
            
        self.update_hyperparameters()
    
    def create_PVDER_NN(self):
        """Create a NN model for PVDER data."""
    
        if self.model_type == 'dense':
            self.PVDER_NN = self.create_dense_NN()
    
        elif self.model_type == 'LSTM':
            self.PVDER_NN = self.create_LSTM_NN()

        return self.PVDER_NN

    def update_hyperparameters(self,num_layers=None,units_per_layer=None,default_units=None,ADD_BATCHNORM=None):
        """Update DER NN model hyperparametres."""
        
        
        if num_layers is not None:
            self._num_layers = num_layers
        else:
            self._num_layers = self._DERmodel_spec[self.model_type]['num_layers']
        
        if units_per_layer is not None:
            self._units_per_layer = units_per_layer
        else:
            self._units_per_layer = self._DERmodel_spec[self.model_type]['num_units']
        
        if default_units is not None:
            self._default_units = default_units
        else:
            self._default_units = self._DERmodel_spec[self.model_type]['default_units']
        
        if ADD_BATCHNORM is not None and self.model_type == 'dense':
            self._ADD_BATCHNORM = ADD_BATCHNORM
        elif self.model_type == 'dense' :
            self._ADD_BATCHNORM = self._DERmodel_spec[self.model_type]['ADD_BATCHNORM']            
        
    def create_dense_NN(self):
        """Create a densely connected NN."""

        assert self._num_layers >=1, 'Number of layers should be greater than or equal to one!'
        
        print("Creating {} NN with following hyperparameters:\nLayers:{}\nUnits:{}".format(self.model_type,self._num_layers,self._units_per_layer))

        #Model #1 — MLP
        PVDER_dense = Sequential() # Initialising the DNN
        PVDER_dense.add(BatchNormalization(input_shape = (self.num_features,),name='batchnorm_input'))

        for layer in range(self._num_layers):

            try:
                units=self._units_per_layer[layer]
            except IndexError:
                units = None

            if units is None:
                units = self.default_units

            PVDER_dense.add(Dense(units = units, activation = 'relu',name='dense_'+str(layer+1)))
            if self._ADD_BATCHNORM:
                PVDER_dense.add(BatchNormalization(name='batchnorm_'+str(layer)))

        PVDER_dense.add(Dense(units = self.num_outputs, activation = 'linear',name='dense_output'))
        PVDER_dense.compile(optimizer = 'adam', loss = 'mse')
        PVDER_dense.summary() 

        return PVDER_dense

    def create_LSTM_NN(self):
        """Create an LSTM NN."""
    
        assert self._num_layers >=1, 'Number of layers should be greater than or equal to one!'

        print("Creating {} NN with following hyperparameters:\nLayers:{}\nUnits:{}".format(self.model_type,self._num_layers,self._units_per_layer))
        
        #Model #2 — LSTM
        PVDER_LSTM = Sequential() # Initialising the DNN

        for layer in range(self._num_layers):

            try:
                units=self._units_per_layer[layer]
            except IndexError:
                units = None

            if units is None:
                units = self.default_units

            if layer == 0:
                PVDER_LSTM.add(LSTM(units =units,input_shape=(None,self.num_features),return_sequences=True,name='LSTM_'+str(layer+1))) # returns a sequence of vectors of dimension units
            else:
                PVDER_LSTM.add(LSTM(units = units,return_sequences=True,name='LSTM_'+str(layer+1)))

        PVDER_LSTM.add(LSTM(self.num_outputs, return_sequences=True,name='LSTM_output'))

        PVDER_LSTM.compile(optimizer = 'adam', loss = 'mse')
        PVDER_LSTM.summary() 
        
        return PVDER_LSTM

    def create_tfdataset(self,batch_size):
        """Create pipeline."""
        
        train_dataset = tf.data.Dataset.from_tensor_slices((self.DERdata.Xtrain,self.DERdata.Ytrain))
        
        # Shuffle and slice the dataset.
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        #train_dataset = dataset.cache()  # small dataset can be entirely cached in RAM, for TPU this is important to get good performance
        
        # Shuffle, repeat, and batch the examples,epeat mandatory for Keras for now,drop_remainder is important on TPU,batch size is fixed
        #train_dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).repeat().batch(batch_size, drop_remainder=True)  
        
        # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
        #train_dataset = dataset.prefetch(-1)  
        
        return train_dataset      
        
    def train_and_evaluate(self,batch_size=32,n_epochs=10,USE_DATSET=True):
        """Train and evaluate."""

        assert isinstance(self.PVDER_NN, Sequential), 'NN model is not of correct type!'
        
        if USE_DATSET:
            train_dataset = self.create_tfdataset(batch_size)
            self.PVDER_NN.fit(train_dataset, epochs=n_epochs)
            
        else:
            self.PVDER_NN.fit(self.DERdata.Xtrain,self.DERdata.Ytrain, validation_split=0.2, batch_size=32,epochs=n_epochs)

        n_evaluate_samples = int(len(self.DERdata.Xtrain)*0.2)
        print('Evaluating on {} samples'.format(n_evaluate_samples))

        self.PVDER_NN.evaluate(self.DERdata.Xtrain[0:n_evaluate_samples,:],self.DERdata.Ytrain[0:n_evaluate_samples,:])
    
    def predict_n_steps(self,n_steps=2):
        """Train and evaluate n time steps."""
        
        assert isinstance(self.PVDER_NN, Sequential), 'NN model is not of correct type!'

        if self.model_type == 'dense':
            Ypredict = self.PVDER_NN.predict(self.DERdata.Xtrain[0:n_steps,:])

        elif self.model_type == 'LSTM':
            Ypredict = self.PVDER_NN.predict(self.DERdata.Xtrain[0:n_steps,:,:])

        return Ypredict
        
    def compare_trajectories(self,n_steps=2):
        """Compare trajectories."""

        Ypredict = self.predict_n_steps(n_steps)

        time_values = np.linspace(0.0, n_steps*0.001, num=n_steps)
        plot_values = Ypredict

        for i in range(3):
            plot_values= Ypredict[:,i]
            plt.plot(time_values,plot_values)

        plt.show()
    
    
    
    
    
 
        
    




