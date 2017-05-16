from callbacks import SnapshotCallback
from datasets import load_dataset
from models import RegressionModel, load_from_config

init_model_config = RegressionModel.create(
    '/tmp/regression_model.keras',
    use_adadelta=True,
    learning_rate=0.001,
    input_shape=(120, 320, 3))

model = load_from_config(init_model_config)

# this path contains a dataset in the prescribed format
dataset = load_dataset('M:\\selfdrive\\SelfDrivingData\\test_out')

# snapshots the model after each epoch
snapshot = SnapshotCallback(
    model,
    snapshot_dir='/tmp/snapshots/',
    score_metric='val_rmse')

model.fit(dataset, {
    'batch_size': 32,
    'epochs': 40,
},
          final=False,  # don't train on the test holdout set
          callbacks=[snapshot])

# save model to local file and return the 'config' so it can be loaded
model_config = model.save('/tmp/regression.keras')

# evaluate the model on the test holdout
print(model.evaluate(dataset))