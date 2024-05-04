"""Defines the configuration to be loaded before running any experiment"""
from configobj import ConfigObj
import string


class Config(object):
    def __init__(self, filename: string):
        """
        Read from a config file
        :param filename: name of the file to read from
        """

        self.filename = filename
        config = ConfigObj(self.filename)
        self.config = config

        # Comments on the experiments running
        self.comment = config["comment"]

        # Model name and location to store
        self.model_path = config["train"]["model_path"]

        # number of training examples
        self.num_train = config["train"].as_int("num_train")
        self.num_val = config["train"].as_int("num_val")
        self.num_test = config["train"].as_int("num_test")
        self.rate_train = config["train"].as_float("rate_train")
        self.rate_val = config["train"].as_float("rate_val")
        self.rate_test = config["train"].as_float("rate_test")
        self.num_points = config["train"].as_int("num_points")
        self.Q_size = config["train"].as_int("Q_size")
        # Weight to the loss function for stretching
        # self.loss_weight = config["train"].as_float("loss_weight")

        # dataset
        # self.dataset_path = config["train"]["dataset"]

        # Number of epochs to run during training
        self.epochs = config["train"].as_int("num_epochs")

        # batch size, based on the GPU memory
        self.batch_size = config["train"].as_int("batch_size")

        # Mode of training, 1: supervised, 2: RL
        self.mode = config["train"].as_int("mode")

        # Learning rate
        # self.lr = config["train"].as_float("lr")

        # self.shape = config["train"]["shape"]

        self.d_scale = config["train"].as_bool("d_scale")
        self.d_mean = config["train"].as_bool("d_mean")
        self.d_rotation = config["train"].as_bool("d_rotation")
        self.if_normals = config["train"].as_bool("if_normals")

        self.more = config["train"]["more"]

        self.last = config["train"]["last"]

        self.dataset_dir = config["train"]["dataset_dir"]
        self.dataset_test_dir = config["train"]["dataset_test_dir"]
        self.test_on_another_dataset = config["train"].as_bool("test_on_another_dataset")

    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and
        what parameters have been used.
        :return:
        """
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config_synthetic.yml")
    print(file.write_config())
