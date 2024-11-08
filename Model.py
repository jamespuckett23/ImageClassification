import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from NetWork import ResNet, resnet18
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = resnet18()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(),  \
                             self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.config.lr_step, gamma=self.config.lr_decay
        )
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        losses = []

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            epoch_loss = 0.0
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Manually update or use scheduler from pytorch
            
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                batch_x = curr_x_train[i * self.config.batch_size: (i + 1) * self.config.batch_size]
                batch_y = curr_y_train[i * self.config.batch_size: (i + 1) * self.config.batch_size]

                batch_x = np.array([parse_record(img, training=True) for img in batch_x])
                batch_x = torch.tensor(batch_x, dtype=torch.float32)
                batch_y = torch.tensor(batch_y, dtype=torch.long)
                
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                    self.network = self.network.cuda()

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.network(batch_x)
                loss = self.loss_fn(outputs, batch_y)

                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            losses.append(avg_epoch_loss)

            # Step the scheduler
            self.scheduler.step()
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))


            if epoch % self.config.save_interval == 0:
                self.save(epoch)
                
        ### YOUR CODE HERE
        #Plot the curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_epoch + 1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
        # plt.savefig("training_loss_curve.png")

        # Directory where you want to save the image
        save_dir = '/content/data'
        os.makedirs(save_dir, exist_ok=True)

        # Define the path for saving the plot
        image_path = os.path.join(save_dir, "training_loss_curve.png")

        # Save the plot before showing it
        plt.savefig(image_path)
        plt.show()

        # Print the path to confirm where it's saved
        print("Plot saved to:", image_path)

        print("Training loss curve saved as training_loss_curve.png")
        ### YOUR CODE HERE


    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                img = parse_record(x[i], training=False)
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

                # Move data to device if available
                if torch.cuda.is_available():
                    img = img.cuda()
                    self.network = self.network.cuda()

                output = self.network(img)
                _, pred = torch.max(output, 1)
                preds.append(pred.item())
                ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            if torch.cuda.is_available():
                y, preds = y.cuda(), preds.cuda()
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))