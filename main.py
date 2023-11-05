from scripts.classes import *

b_neurons = [
		'AVAR',
		'AVAL',
		'SMDVR',
		'SMDVL',
		'SMDDR',
		'SMDDL',
		'RIBR',
		'RIBL'
	]

test = False
if test:
	worm_num = 4

	data = Database(worm_num, verbose=0)
	data.exclude_neurons(b_neurons)
	loaded_vs = data.loadBundleVisualizer()
	loaded_vs.plot3D_mapping(show_legend=True)
	exit()

### Load Data (excluding behavioural neurons) and plot
for worm_num in range(5):
	data = Database(worm_num, verbose=1)
	data.exclude_neurons(b_neurons)
	vs = data.createVisualizer()
	#vs.plotting_neuronal_behavioural()
	vs.plot3D_mapping(show_legend=True, grid_off=True)

	### Preprocess and prepare data for BundLe Net
	bundle_model = vs.attachBundleNet(epochs=2000)
	vs.plot_loss()
	vs.plot3D_mapping(show_legend=True)
	#vs.model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))

	#bundle_model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))

	#transformed_output = bundle_model.tau(vs.X)  # same as transformed_output = bundle_model.tau.predict(vs.X)
exit()

### Comparable Embeddings
bundle_model.load_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
weights_T_Y = bundle_model.T_Y.get_weights()
weights_predictor = bundle_model.predictor.get_weights()
