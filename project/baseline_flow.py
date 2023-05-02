from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 1 if row.rating >= 4 else 0

class BaselineNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        import numpy as np

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({'label': labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        np.random.seed(0)
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.traindf.shape[0]}')
        print(f'num of rows in validation set: {self.valdf.shape[0]}')

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.ensemble import RandomForestClassifier

        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        import numpy as np
        np.random.seed(0)
        
        # Random binary predictions, equal probability
        # self.preds = np.random.randint(0, 2, size=len(self.valdf))
        # 2023-05-02 05:05:21.110 [46/end/217 (pid 24183)] Baseline Accuracy: 0.502
        # 2023-05-02 05:05:23.714 [46/end/217 (pid 24183)] Baseline AUC: 0.493

        # Binary predictions, weighted by train set label likelihood
        pos_prob = self.traindf['label'].sum() / len(self.traindf)
        self.preds = np.random.choice([0, 1], size=len(self.valdf), p=[1 - pos_prob, pos_prob])
        # 2023-05-02 05:06:19.604 [47/end/221 (pid 24594)] Baseline Accuracy: 0.649
        # 2023-05-02 05:06:22.146 [47/end/221 (pid 24594)] Baseline AUC: 0.511

        self.base_acc = accuracy_score(self.valdf['label'], self.preds)
        self.base_rocauc = roc_auc_score(self.valdf['label'], self.preds)

        self.next(self.end)

    @card(type='corise') # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):

        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))

        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0. 
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        
        current.card.append(Markdown("## Examples of False Positives"))
        false_positives = (self.preds == 1) & (self.valdf['label'].values == 0)
        
        current.card.append(Markdown('Fraction of false positives'))
        current.card.append(Artifact(false_positives.sum() / len(self.valdf)))
        
        valdf_fp = self.valdf[false_positives]
        current.card.append(Table.from_dataframe(valdf_fp[['review']].sample(5)))
        
        
        # TODO: compute the false positive predictions where the baseline is 0 and the valdf label is 1. 
        # TODO: display the false_negatives dataframe using metaflow.cards
        current.card.append(Markdown("## Examples of False Negatives"))
        false_negatives = (self.preds == 0) & (self.valdf['label'].values == 1)
        
        current.card.append(Markdown('Fraction of false negatives'))
        current.card.append(Artifact(false_negatives.sum() / len(self.valdf)))
        
        valdf_fn = self.valdf[false_negatives]
        current.card.append(Table.from_dataframe(valdf_fn[['review']].sample(5)))


if __name__ == '__main__':
    BaselineNLPFlow()
