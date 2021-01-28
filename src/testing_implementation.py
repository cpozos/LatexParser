
from data import DataBuilder
from utilities.persistance import load_test_data
from utilities.tensor import show_tensor_as_image

data_builder = DataBuilder()
test_dataset = data_builder.get_dataset_for('test')

results_path = "" #Assign the path where the results were saved
data = load_test_data(results_path)
targets = data[0]
predictions = data[1]
img_tensor = test_dataset.get_image_tensor(targets[0])

show_tensor_as_image(img_tensor)