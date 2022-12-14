from .nmr import NMRModel, get_df_from_outcars
from .dflow_adapter import dflow_task
from typing import List
from fire import Fire
import os
import pandas as pd
from ase.io import read


DEFAULT_SOAP_EXPRESSION = "soap cutoff=5.5 cutoff_transition_width=0.5 n_max=9 l_max=9 atom_sigma=0.55 n_Z=1 n_species=4 species_Z={11, 12, 25, 8}"
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 500


class Ai2Nmr:

    def __init__(self, elements: List[str], soap_expr=DEFAULT_SOAP_EXPRESSION):
        self._elements = elements
        self._soap_expr = soap_expr


    def train_model(self, outcar_folders=None, outcar_folders_dir=None, out_dir='./out/',
                    test_size=0.2, train_size=None,
                    random_state=None, shuffle=True, stratify=None,
                    epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
        os.makedirs(out_dir, exist_ok=True)

        if not outcar_folders:
            assert outcar_folders_dir, 'one of outcar_folders or outcar_folders_dir must be set'
            outcar_folders = [os.path.join(outcar_folders_dir, folder)  for folder in os.listdir(outcar_folders_dir)]

        dfs = pd.DataFrame()
        for folder in outcar_folders:
            outcar_list = os.listdir(folder)
            outcar_list.sort(key=lambda x: int(x.lstrip('OUTCAR')))
            df = get_df_from_outcars([os.path.join(folder, outcar)
                                      for outcar in outcar_list], expression=self._soap_expr, elements=self._elements)
            dfs = pd.concat([dfs, df])

        X, y = NMRModel.get_Xy_from_df(dfs)
        history = NMRModel(X, y).train_model(out_dir=out_dir,
                                             test_size=test_size, train_size=train_size,
                                             random_state=random_state, shuffle=shuffle, stratify=stratify,
                                             epochs=epochs, batch_size=batch_size)


    def predict(self, traj_path: str, model='./out/model/'):
        test_atoms = read(traj_path)
        nmr = NMRModel()
        nmr.load_model(model)
        return nmr.predict_fcshifts(test_atoms, expression=self._soap_expr, elements=self._elements)


def main():
    Fire(Ai2Nmr)

if __name__ == '__main__':
    main()