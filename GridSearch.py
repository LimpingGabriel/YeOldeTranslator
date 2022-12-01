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
        "batch_size": [32],
        "num_layers": [6],
        "d_model": [256, 128],
        "dff": [1024, 512, 256],
        "num_heads": [8],
        "dropout": [0.3],
        "epochs" : [200]
        }

    
    results = []
    f=open("GridSearch.txt", "a")
    settings.logger.debug("Opened GridSearch.txt.")

    options = list((dict(zip(all_parameters.keys(), values)) for values in product(*all_parameters.values())))
    i = 1
    for parameters in options:
        settings.logger.info("Using hyperparameters " + str(parameters))
        settings.logger.info("Model {}/{}.".format(i, len(options)))
        out = train_model(dataset, parameters)
        parameters["val_loss"] = out["val_loss"]
        parameters["val_acc"] = out["val_acc"]
        parameters["val_bleu"] = out["val_bleu"]

        settings.logger.info(str(parameters))
        results.append(parameters)

        f.write(str(parameters))
        f.write("\n")
        f.flush()
        i += 1

    f.close()
