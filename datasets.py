import os
from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import pickle as pkl
import numpy as np

import cv2

class MnistMetaEnv:
    def __init__(self, height=28, length=28):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.MNIST(root='./data', train=True, download=True)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.height, self.length))
        self.norm = transforms.Normalize((0.5,), (0.5,))
        
        self.sample_training_task()

    def sample_training_task(self, batch_size=64):
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.norm(self.to_tensor(self.resize(self.data[idx][0]))).numpy() for idx in task_idx]), dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.norm(self.to_tensor(self.resize(self.data[idx][0]))).numpy() for idx in task_idx]), dtype=torch.float)
        return batch, task

    def make_tasks(self):
        self.task_to_examples = {}
        self.all_tasks = set(self.data.train_labels.numpy())
        for i, digit in enumerate(self.data.train_labels.numpy()):
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def split_validation_and_training_task(self):
        self.validation_task = {9}
        self.training_task = self.all_tasks - self.validation_task


class OmniglotMetaEnv:
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.Omniglot(root='./data', download=True)
        self.make_tasks()
        self.split_validation_and_training_task()
        self.resize = transforms.Resize((self.height, self.length))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5,), (0.5,))

    def sample_training_task(self, batch_size=4):
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.norm(self.to_tensor(self.resize(self.data[idx][0]))).numpy() for idx in task_idx]), dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=64):
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.norm(self.to_tensor(self.resize(self.data[idx][0]))).numpy() for idx in task_idx]), dtype=torch.float)
        return batch, task

    def make_tasks(self):
        self.task_to_examples = {}
        self.all_tasks = set()
        for i, (_, digit) in enumerate(self.data):
            self.all_tasks.update([digit])
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 20))
        self.training_task = self.all_tasks - self.validation_task

class VggFaceMetaEnv(Dataset):
    def __init__(self, height=32, length=32):
        self.channels = 3
        self.height = height
        self.length = length
        self.resize = transforms.Resize((height, length))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if os.path.exists(f"./data/vgg/vgg_tasks_{self.height}_norm_0point5.pkl"):
            self.tasks = pkl.load(open(f"./data/vgg/vgg_tasks_{self.height}_norm_0point5.pkl", "rb"))
            self.validation_task = pkl.load(open(f"./data/vgg/vgg_validation_tasks_{self.height}.pkl", "rb"))
            self.training_task = pkl.load(open(f"./data/vgg/vgg_training_tasks_{self.height}.pkl", "rb"))
            print("Pickle data exits")
        
        else:
            self.tasks = self.get_tasks()
            self.all_tasks = set(self.tasks)
            self.split_validation_and_traning_task()
    
    def get_tasks(self):

        tasks_temp = pkl.load(open(f"vgg_tasks.pkl", "rb"))
        tasks = dict()

        for task in range(tasks_temp.shape[0]):
            tasks[task] = []
            for img_idx in range(tasks_temp.shape[1]):
                tasks_temp_rgb = cv2.cvtColor(tasks_temp[task, img_idx, :, :, :], cv2.COLOR_BGR2RGB)
                img = Image.fromarray((tasks_temp_rgb*255).astype(np.uint8))
                tasks[task].append(np.array(self.norm(self.to_tensor(self.resize(img)))))
            tasks[task] = np.array(tasks[task])
        pkl.dump(tasks, open(f"./data/vgg/vgg_tasks_{self.height}_norm_0point5.pkl", 'wb'))
        return tasks
    
    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 20))
        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open(f"./data/vgg/vgg_validation_tasks_{self.height}.pkl", 'wb'))
        pkl.dump(self.training_task, open(f"./data/vgg/vgg_training_tasks_{self.height}.pkl", 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def __len__(self):
        return len(self.files)

class miniImageNetMetaEnv(Dataset):
    def __init__(self, height=32, length=32):
        self.channels = 3
        self.height = height
        self.length = length
        self.resize = transforms.Resize((height, length))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if os.path.exists(f"./data/miniimagenet/mini_tasks_{self.height}_norm_0point5.pkl"):
            self.tasks = pkl.load(open(f"./data/miniimagenet/mini_tasks_{self.height}_norm_0point5.pkl", "rb"))
            self.validation_task = sorted(pkl.load(open(f"./data/miniimagenet/mini_validation_tasks_{self.height}.pkl", "rb")))
            self.training_task = sorted(pkl.load(open(f"./data/miniimagenet/mini_training_tasks_{self.height}.pkl", "rb")))
            print("Pickle data exits")
        else:
            self.tasks = self.get_tasks()
            self.all_tasks = set(self.tasks)
            self.split_validation_and_training_task()

    def get_tasks(self):

        tasks = dict()
        path = 'miniImageNet/data'
        for task in os.listdir(path):
            tasks[task] = []
            task_path = os.path.join(path, task)
            for imgs in os.listdir(task_path):
                img = Image.open(os.path.join(task_path, imgs))
                tasks[task].append(np.array(self.norm(self.to_tensor(self.resize(img)))))
            tasks[task] = np.array(tasks[task])
        pkl.dump(tasks, open(f"./data/miniimagenet/mini_tasks_{self.height}_norm_0point5.pkl", 'wb'))
        return tasks

    def split_validation_and_training_task(self):
        self.validation_task = set(random.sample(self.all_tasks, 20))
        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open(f"./data/miniimagenet/mini_validation_tasks_{self.height}.pkl", 'wb'))
        pkl.dump(self.training_task, open(f"./data/miniimagenet/mini_training_tasks_{self.height}.pkl", 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float)
        return batch, task

    def __len__(self):
        return len(self.files)