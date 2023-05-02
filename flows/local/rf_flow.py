from metaflow import FlowSpec, step, card


class RF_Flow(FlowSpec):
    """
    train a random forest
    """

    @step
    def start(self):
        self.next(self.load_data, self.rf_model)

    @step
    def load_data(self):
        """
        Load the data
        """
        # Import scikit-learn dataset library
        from sklearn import datasets

        # Load dataset
        self.iris = datasets.load_iris()
        self.X = self.iris["data"]
        self.y = self.iris["target"]
        self.next(self.join)

    @step
    def rf_model(self):
        """
        build random forest model
        """
        from sklearn.ensemble import RandomForestClassifier

        self.clf = RandomForestClassifier(
            n_estimators=10, max_depth=None, min_samples_split=2, random_state=0
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.X = inputs.load_data.X
        self.y = inputs.load_data.y
        self.clf = inputs.rf_model.clf
        self.next(self.train)

    @step
    def train(self):
        """
        Train the model
        """
        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.clf, self.X, self.y, cv=5)
        print(self.scores)
        self.next(self.end)

    @step
    def end(self):
        """
        End of flow!
        """
        print(self.scores)
        print("RF_Flow is all done.")


if __name__ == "__main__":
    RF_Flow()
