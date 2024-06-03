import ml_collections

def get_config():

  config = ml_collections.ConfigDict()


  # wandb
  config.wandb = wandb = ml_collections.ConfigDict()
  wandb.entity = None # team name, must have already created
  wandb.project = "ddpm-flax-fashion-mnist"  # required filed if use W&B logging
  wandb.job_type = "training"
  wandb.name = None # run name, optional
  wandb.log_train = None # log training metrics 
  wandb.log_sample = None # log generated samples to W&B
  wandb.log_model = None # log final model checkpoint as W&B artifact
  

  # training
  config.training = training = ml_collections.ConfigDict()
  training.num_train_steps = 9375
  training.num_train_steps_fin = 1000 #only for transfer learning
  training.transfer_learning= False
  training.loss_type = 'l2'
  training.half_precision = False
  training.save_and_sample_every = 1000
  training.num_sample = 70

  

  # ema
  config.ema = ema = ml_collections.ConfigDict()
  ema.beta = 0.995
  ema.update_every = 10
  ema.update_after_step = 100
  ema.inv_gamma = 1.0
  ema.power = 2 / 3
  ema.min_value = 0.0
 

  # ddpm 
  config.ddpm = ddpm = ml_collections.ConfigDict()
  ddpm.beta_schedule = 'cosine'
  ddpm.timesteps = 1000
  ddpm.p2_loss_weight_gamma = 1. # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
  ddpm.p2_loss_weight_k = 1
  ddpm.self_condition = False # not tested yet
  ddpm.pred_x0 = False # by default, the model will predict noise, if True predict x0


  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'mnist'
  data.batch_size = 128
  data.cache = False
  data.image_size = 28
  data.channels = 1
  


  # model
  config.model = model = ml_collections.ConfigDict()
  model.dim = 10
  model.dim_mults = (1, 2,3, 4)
  model.number_quantum_channel= 1
  model.model = 'UNet' # 'QVUNet' or 'FullQVUNet' or 'QuanvUNet' or 'UNet'
  model.quantum_channel_vertex = 1
  model.name_ansatz_vertex = 'HQConv_ansatz' #'HQConv_ansatz' or 'FQConv_ansatz'
  model.num_layer_vertex = 3
  model.quantum_channel_quan = 1
  model.name_ansatz_quan = 'HQConv_ansatz' #  'HQConv_ansatz' or 'FQConv_ansatz
  model.num_layer_quan = 3



  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'Adam'
  optim.lr = 1e-3
  optim.beta1 = 0.9
  optim.beta2 = 0.99
  optim.eps = 1e-8

  config.seed = 42

  return config
