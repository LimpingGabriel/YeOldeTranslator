from itertools import product

from train import train_model
from TextDataset import *
from GlobalSettings import GlobalSettings

if __name__ == "__main__":
    settings = GlobalSettings()
    dataset = TextDataset("Shakescleare")
    dataset.prepare()
    dataset.make_split(0.98)

    all_parameters = {
        "batch_size": [256, 16, 32, 64],
        "num_layers": [1, 2, 4, 6],
        "d_model": [1, 32, 64, 128, 256, 512],
        "dff": [1, 64, 128, 256, 512, 1024, 2048],
        "num_heads": [1, 2, 4, 8],
        "dropout": [0.1],
        "epochs" : [1, 300]
        }

    
    results = []
    f=open("GridSearch.txt", "a")
    settings.logger.debug("Opened GridSearch.txt.")

    for parameters in (dict(zip(all_parameters.keys(), values)) for values in product(*all_parameters.values())):
        settings.logger.info("Using hyperparameters " + str(parameters))
        out = train_model(dataset, parameters)
        parameters["val_loss"] = out["val_loss"]
        parameters["val_acc"] = out["val_acc"]
        parameters["val_bleu"] = out["val_bleu"]

        settings.logger.info(str(parameters))
        results.append(parameters)

        f.write(str(parameters))
        f.flush()

    f.close()