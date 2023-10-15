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
  
        # dataset path
        self.dataset_path = config["train"]["dataset_path"]
        self.dataset_path_separately = config["train"]["dataset_path_separately"]

        # Number of epochs to run during training
        self.epochs = config["train"].as_int("num_epochs")

        # Batch size, based on the GPU memory
        self.batch_size = config["train"].as_int("batch_size")

        # Mode of training, 1: supervised, 2: RL
        self.mode = config["train"].as_int("mode")

        # Learning rate
        self.lr = config["train"].as_float("lr")

        # Normalization options
        self.d_scale = config["train"].as_bool("d_scale")
        self.d_mean = config["train"].as_bool("d_mean")

        # Last layer of fitting network
        self.last = config["train"]["last"]

        # Loss function weights
        self.lamb_0_0 = config["train"].as_float("lamb_0_0")
        self.lamb_0_1 = config["train"].as_float("lamb_0_1")
        self.lamb_0_2 = config["train"].as_float("lamb_0_2")
        self.lamb_0_3 = config["train"].as_float("lamb_0_3")
        self.lamb_0_4 = config["train"].as_float("lamb_0_4")
        self.lamb_0_5 = config["train"].as_float("lamb_0_5")
        self.lamb_0_6 = config["train"].as_float("lamb_0_6")
        self.lamb_1 = config["train"].as_float("lamb_1")

        # number of primitives
        self.num_primitives = config["train"].as_int("num_primitives")

        # number of iterations for clustering
        self.cluster_iterations = config["train"].as_int("cluster_iterations")

        # pretrain model path
        self.detection_model_path = config["train"]["detection_model_path"]
        self.fitting_model_path = config["train"]["fitting_model_path"]

        # normal options
        self.if_detection_normals = config["train"].as_bool("if_detection_normals")
        self.if_fitting_normals = [int(i) for i in config["train"]["if_fitting_normals"]]

        # knn options
        self.knn = config["train"].as_int("knn")
        self.knn_step = config["train"].as_int("knn_step")

        # Notes
        self.more = config["train"]["more"]


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
