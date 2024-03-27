import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.data


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)

class ECPICK(nn.Module):
    def __init__(self, output_classes, cuda_support=False, cuda_device=None, dropout_rate=0.8, relu_size=384, beta=0.6):
        super(ECPICK, self).__init__()

        self.dropout_rate = dropout_rate
        self.relu_size = relu_size
        self.beta = beta
        self.hierarchical_level = len(output_classes)

        momentum = 0.99
        eps = 1e-01

        # region [ CNN Encoding Layers ]
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(4, 20), stride=1, dilation=1),  # (128 x 997 x 1)
            nn.ReLU(),  # (128 x 997 x 1)
            nn.MaxPool2d(kernel_size=(997, 1)),  # (128 x 1 x 1)
            Flatten(),  # (128)
            nn.BatchNorm1d(128, momentum=momentum, eps=eps)  # (128)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(8, 20), stride=1, dilation=1),  # (128 x 993 x 1)
            nn.ReLU(),  # (128 x 993 x 1)
            nn.MaxPool2d(kernel_size=(993, 1)),  # (128 x 1 x 1)
            Flatten(),  # (128)
            nn.BatchNorm1d(128, momentum=momentum, eps=eps)  # (128)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(16, 20), stride=1, dilation=1),  # (128 x 985 x 1)
            nn.ReLU(),  # (128 x 985 x 1)
            nn.MaxPool2d(kernel_size=(985, 1)),  # (128 x 1 x 1)
            Flatten(),  # (128)
            nn.BatchNorm1d(128, momentum=momentum, eps=eps)  # (128)
        )

        if cuda_support:
            self.cnn1.cuda(device=cuda_device)
            self.cnn2.cuda(device=cuda_device)
            self.cnn3.cuda(device=cuda_device)

        input_feature = 384
        # endregion

        # region [ Hierarchical Layers ]

        # region [ Global Flow Definition ]
        self.global_hidden_layers = []
        self.global_output_layer = None
        self.global_output_size = 0
        self.global_final_output = None

        for i in range(len(output_classes)):
            input_size = input_feature if i == 0 else input_feature + relu_size
            self.global_output_size += len(output_classes[i])

            global_hidden_layer = nn.Sequential(nn.Linear(input_size, relu_size),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(relu_size, momentum=momentum, eps=eps),
                                                nn.Dropout(p=dropout_rate))
            if cuda_support:
                global_hidden_layer.cuda(device=cuda_device)

            self.global_hidden_layers.append(global_hidden_layer)

        self.global_hidden_layers = nn.ModuleList(self.global_hidden_layers)
        if cuda_support:
            self.global_hidden_layers.cuda(device=cuda_device)
        self.global_output_layer = nn.Sequential(nn.Linear(input_feature + relu_size, self.global_output_size), nn.Sigmoid())
        if cuda_support:
            self.global_output_layer.cuda(device=cuda_device)
        # endregion

        # region [ Local Flow Definition ]
        self.local_hidden_layers = []
        self.local_output_layers = []
        self.local_output_index = []
        self.local_output = []
        self.local_final_output = None

        for i in range(len(output_classes)):
            if i == 0:
                input_size = input_feature + relu_size
            else:
                input_size = input_feature + relu_size + len(output_classes[i - 1])

            local_hidden_layer = nn.Sequential(nn.Linear(input_size, relu_size),
                                               nn.ReLU(),
                                               nn.BatchNorm1d(relu_size, momentum=momentum, eps=eps),
                                               nn.Dropout(p=dropout_rate))
            if cuda_support:
                local_hidden_layer.cuda(device=cuda_device)
            self.local_hidden_layers.append(local_hidden_layer)

            output_size = len(output_classes[i])
            local_output_layer = nn.Sequential(nn.Linear(relu_size, output_size), nn.Sigmoid())
            if cuda_support:
                local_output_layer.cuda(device=cuda_device)
            self.local_output_layers.append(local_output_layer)

            if i == 0:
                self.local_output_index.append((0, len(output_classes[i])))
            else:
                start_cursor = self.local_output_index[i - 1][1]
                self.local_output_index.append((start_cursor, start_cursor + len(output_classes[i])))

        self.local_hidden_layers = nn.ModuleList(self.local_hidden_layers)
        if cuda_support:
            self.local_hidden_layers.cuda(device=cuda_device)
        self.local_output_layers = nn.ModuleList(self.local_output_layers)
        if cuda_support:
            self.local_output_layers.cuda(device=cuda_device)
        # endregion

        # endregion

    def forward(self, x):
        # CNN
        input_feature = torch.cat([self.cnn1(x), self.cnn2(x), self.cnn3(x)], dim=1)

        # region [ Hierarchical Global Flow ]
        global_hidden_layer_outputs = []

        # Compute hidden layer
        for i in range(len(self.global_hidden_layers)):
            # Make Input Value
            if i == 0:
                input_value = input_feature
            else:
                input_value = torch.cat((input_feature, global_hidden_layer_outputs[-1]), dim=1)

            # Compute Global layer output
            global_hidden_layer_outputs.append(self.global_hidden_layers[i](input_value))

        # Compute output layer
        input_value = torch.cat((input_feature, global_hidden_layer_outputs[-1]), dim=1)
        self.global_final_output = self.global_output_layer(input_value)
        # endregion

        # region [ Hierarchical Local Flow ]
        local_hidden_layer_outputs = []
        local_output_layer_outputs = []

        self.local_output = []

        for i in range(len(self.local_hidden_layers)):
            # Make Input Value
            if i == 0:
                input_value = torch.cat([input_feature, global_hidden_layer_outputs[i]], dim=1)
            else:
                input_value = torch.cat([input_feature, global_hidden_layer_outputs[i], local_output_layer_outputs[-1]],
                                        dim=1)

            # Compute Hidden layer output
            local_hidden_layer_outputs.append(self.local_hidden_layers[i](input_value))

            # Compute Output layer output
            local_output_layer_outputs.append(self.local_output_layers[i](local_hidden_layer_outputs[-1]))
            self.local_output.append(local_output_layer_outputs[-1])

        self.local_final_output = torch.cat(self.local_output, dim=1)
        # endregion

        index = self.local_output_index[-1]
        final_output = (self.beta * self.global_final_output) + ((1 - self.beta) * self.local_final_output)

        return {
            'local_output': self.local_final_output,
            'global_output': self.global_final_output,
            'final_output': final_output[:, index[0]:index[1]],
            'final_output_all': final_output
        }


import torch
from torch.nn import BCELoss
from plants_sm.models.lightning_model import InternalLightningModule
from plants_sm.models.fc.fc import DNN
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ECPICKModule(InternalLightningModule):

    def __init__(self, input_dim, layers, classification_neurons, metric=None, learning_rate = 1e-3, layers_to_freeze=0) -> None:

        super().__init__(metric=metric)
        self.layers = layers
        self.classification_neurons = classification_neurons
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.layers_to_freeze = layers_to_freeze
        self.model = self._create_model()

    def _create_model(self):
        self.ecpick_model = ECPICK(self.classification_neurons, cuda_support=False, cuda_device="cuda:0")
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': self.fc_model.parameters()}], lr=self.learning_rate)

        # Define a custom learning rate scheduler using LambdaLR
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'), 'monitor': 'val_loss'}

        return [optimizer], [scheduler]

    def forward(self, x):
        return self.fc_model(x)

    def compute_loss(self, logits, y):

        # Compute Global Loss
        global_loss = func.binary_cross_entropy(logits["global_output"], y, reduction="sum")

        # Compute Local Loss
        local_loss = func.binary_cross_entropy(logits["local_output"], y, reduction="sum")

        # Compute Final Loss
        final_loss = global_loss + local_loss

        return final_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        if not isinstance(x, list):
            x = [x]
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        self.training_step_outputs.append(logits["final_output"])
        self.training_step_y_true.append(y)
        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        if not isinstance(inputs, list):
            inputs = [inputs]
        output = self(inputs)

        self.validation_step_outputs.append(output["final_output"])
        self.validation_step_y_true.append(target)
