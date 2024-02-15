import unittest
from unittest.mock import MagicMock
from ncmcm.classes import *
from sklearn.decomposition import PCA


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.db = Database(neuron_traces=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                           behavior=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           neuron_names=['neuron1', 'neuron2', 'neuron3', 'neuron4'],
                           fps=1)

    def test_init(self):
        db_default = Database(neuron_traces=[[1, 2], [3, 4]], behavior=[0, 1])
        self.assertTrue(np.array_equal(db_default.neuron_traces, np.array([[1, 2], [3, 4]])))
        self.assertIsNone(db_default.fps)
        self.assertEqual(db_default.name, 'nc-mcm')
        self.assertTrue(np.array_equal(db_default.B, np.array([0, 1])))
        self.assertIsNotNone(db_default.states)
        self.assertTrue(np.array_equal(db_default.neuron_names, np.array(['0', '1'])))
        self.assertIsNone(db_default.pred_model)

        # Example with custom parameters
        neuron_traces_custom = np.array([[1, 2], [4, 5]])
        behavior_custom = np.array([1, 0])
        neuron_names_custom = np.array(['neuron1', 'neuron2'])
        states_custom = np.array(['state0', 'state1'])
        fps_custom = 30.0
        name_custom = 'custom-name'

        db_custom = Database(
            neuron_traces=neuron_traces_custom,
            behavior=behavior_custom,
            neuron_names=neuron_names_custom,
            states=states_custom,
            fps=fps_custom,
            name=name_custom
        )

        self.assertTrue(np.array_equal(db_custom.neuron_traces, neuron_traces_custom))
        self.assertEqual(db_custom.fps, fps_custom)
        self.assertEqual(db_custom.name, name_custom)
        self.assertTrue(np.array_equal(db_custom.B, behavior_custom))
        self.assertTrue(np.array_equal(db_custom.states, states_custom))
        self.assertTrue(np.array_equal(db_custom.neuron_names, neuron_names_custom))
        self.assertIsNone(db_custom.pred_model)

    def test_exclude_neurons(self):
        # Exclude an existing neuron
        self.db.exclude_neurons(['neuron1'])
        self.assertTrue(np.array_equal(self.db.neuron_traces, np.array([[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])))
        self.assertTrue(np.array_equal(self.db.neuron_names, np.array(['neuron2', 'neuron3', 'neuron4'])))
        # Exclude a non-existing neuron
        self.db.exclude_neurons(['nonexistent_neuron'])
        self.assertTrue(np.array_equal(self.db.neuron_traces, np.array([[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                                                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])))
        self.assertTrue(np.array_equal(self.db.neuron_names, np.array(['neuron2', 'neuron3', 'neuron4'])))

    def test_createVisualizer(self):
        # Without mapping
        visualizer_no_mapping = self.db.createVisualizer(window=3, epochs=20)
        self.assertIsNotNone(visualizer_no_mapping)
        self.assertIsNotNone(visualizer_no_mapping.mapping)
        self.assertIsNotNone(visualizer_no_mapping.tau_model)
        # With PCA mapping
        mapping_pca = PCA(n_components=2)
        visualizer_with_mapping = self.db.createVisualizer(mapping=mapping_pca)
        self.assertIsNotNone(visualizer_with_mapping)
        self.assertEqual(visualizer_with_mapping.mapping, mapping_pca)
        self.assertIsNone(visualizer_with_mapping.tau_model)

    def test_fit_model(self):
        # Create a MagicMock object to replace the base_model
        base_model_mock = MagicMock()
        base_model_mock.fit.side_effect = lambda x, y: base_model_mock  # Mocking the fit method
        base_model_mock.predict.side_effect = lambda x: np.zeros(x.shape[0])  # Mocking the predict method
        base_model_mock.predict_proba.side_effect = lambda x: np.zeros(len(self.db.states))  # Mocking predict_proba
        # Call the fit_model without CustomModel activated
        result = self.db.fit_model(base_model_mock, prob_map=True, binary=False)
        self.assertTrue(result)  # Assuming fit_model returns True on success
        self.assertTrue(hasattr(self.db, 'pred_model'))  # Check if pred_model attribute is set
        self.assertTrue(hasattr(self.db, 'B_pred'))  # Check if B_pred attribute is set
        self.assertTrue(hasattr(self.db, 'yp_map'))  # Check if yp_map attribute is set

    def test_cluster_BPT(self):
        self.db.yp_map = np.zeros((self.db.neuron_traces.shape[0], len(np.unique(self.db.B))))
        result = self.db.cluster_BPT(nrep=5,
                       max_clusters=2,
                       sim_m=10,
                       sim_s=10,
                       chunks=2,
                       kmeans_init='auto',
                       plot_markov=False)

        self.assertTrue(result)
        self.assertTrue(hasattr(self.db, 'p_memoryless'))  # Check if pred_model attribute is set
        self.assertTrue(hasattr(self.db, 'xc'))  # Check if pred_model attribute is set

        self.db.yp_map = None
        result = self.db.cluster_BPT(nrep=5,
                                max_clusters=2,
                                sim_m=10,
                                sim_s=10,
                                chunks=2,
                                kmeans_init='auto',
                                plot_markov=False)
        self.assertFalse(result)

    def test_step_plot(self, clusters=5):
        # Mock the necessary methods for fit_model and cluster_BPT
        self.db.fit_model = MagicMock(return_value=True)
        self.db.cluster_BPT = MagicMock(return_value=True)
        self.db.yp_map = np.zeros((4, 10))
        self.db.xc = np.array([[np.random.choice(list(range(c+1)), size=10) for c in range(clusters)] for _ in range(10)])
        self.db.p_memoryless = np.random.rand(clusters, 10)

        # Call the step_plot method with mock data
        result = self.db.step_plot(clusters=clusters, nrep=10, sim_m=300, sim_s=300, save=False, show=False)
        self.assertTrue(result)  # Assuming step_plot returns True on success

    def test_behavioral_state_diagram(self, clusters=3):
        # Call the behavioral_state_diagram without p_memoryless
        result = self.db.behavioral_state_diagram(cog_stat_num=3, offset=2.5, adj_matrix=True,
                                                  show=False, save=True, interactive=False)
        self.assertFalse(result)  # Assuming behavioral_state_diagram returns True on success

        # Call the behavioral_state_diagram with p_memoryless
        self.db.p_memoryless = np.random.rand(clusters, 10)
        self.db.xc = np.array([[np.random.choice(list(range(c+1)), size=10) for c in range(clusters)] for _ in range(10)])
        result = self.db.behavioral_state_diagram(cog_stat_num=clusters, offset=2.5, adj_matrix=True,
                                                  show=False, save=False, interactive=False)
        self.assertTrue(result)  # Assuming behavioral_state_diagram returns True on success

    def tearDown(self):
        # Clean up any resources or configurations used in the tests
        pass


if __name__ == '__main__':
    unittest.main()
