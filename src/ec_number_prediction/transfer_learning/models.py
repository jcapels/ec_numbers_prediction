
import torch
from torch.nn import BCELoss
from plants_sm.models.lightning_model import InternalLightningModule
from plants_sm.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ModelECNumber(InternalLightningModule):

    def __init__(self, input_dim, layers, classification_neurons, metric=None, learning_rate = 1e-3, layers_to_freeze=0, 
                 scheduler = False) -> None:

        super().__init__(metric=metric)
        self.layers = layers
        self.classification_neurons = classification_neurons
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.layers_to_freeze = layers_to_freeze
        self.scheduler = scheduler
        self._create_model()
        self._update_constructor_parameters()

    def _create_model(self):
        self.fc_model = DNN(self.input_dim, self.layers, self.classification_neurons, batch_norm=True, last_sigmoid=True, 
                            dropout=None, layers_to_freeze=self.layers_to_freeze)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.fc_model.parameters()}], lr=self.learning_rate)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        return self.fc_model(x)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)


class FineTuneModelECNumber(InternalLightningModule):

    def __init__(self, input_dim, additional_layers, classification_neurons, path_to_model, metric=None, 
                 layers_to_freeze=1, learning_rate = 1e-3, base_layers=[2560], scheduler = True) -> None:

        self.additional_layers = additional_layers
        self.classification_neurons = classification_neurons
        self.path_to_model = path_to_model
        self.layers_to_freeze = layers_to_freeze
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.base_layers = base_layers
        self.scheduler = scheduler
        super().__init__(metric=metric)
        self._create_model()
        

    def _update_constructor_parameters(self):
        self._contructor_parameters.update({"path_to_model": self.path_to_model, "additional_layers": self.additional_layers, 
                                            "classification_neurons": self.classification_neurons, "layers_to_freeze": self.layers_to_freeze, 
                                            "input_dim": self.input_dim, "learning_rate": self.learning_rate, "base_layers": self.base_layers,
                                            "scheduler": self.scheduler})

    def _create_model(self):
        layers = self.base_layers + self.additional_layers
        self.fc_model = DNN(self.input_dim, layers, self.classification_neurons, batch_norm=True, last_sigmoid=True, 
                            dropout=None, layers_to_freeze=self.layers_to_freeze)
        
        # load weigths from pre-trained model

        # 1. filter out unnecessary keys
        model_dict = self.fc_model.state_dict()
        pretrained_dict = torch.load(self.path_to_model)
        if ".ckpt" in self.path_to_model:
            pretrained_dict = {k.replace("fc_model.", ""): v for k, v in pretrained_dict["state_dict"].items() if k.replace("fc_model.", "") in model_dict}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        del pretrained_dict['fc_final.weight']
        del pretrained_dict['fc_final.bias']
        del pretrained_dict['final_batch_norm.weight']
        del pretrained_dict['final_batch_norm.bias']
        del pretrained_dict['final_batch_norm.running_mean']
        del pretrained_dict['final_batch_norm.running_var']
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.fc_model.load_state_dict(model_dict)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.fc_model.parameters()}], lr=self.learning_rate)

        # Define a custom learning rate scheduler using LambdaLR
        if self.scheduler:
            scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}
            return [optimizer], [scheduler]
        else:
            return optimizer

    def forward(self, x):
        return self.fc_model(x)

    def compute_loss(self, logits, y):
        return BCELoss()(logits, y)

    
