import os
import shutil
import json
from datasets import Dataset, DatasetDict, load_from_disk
from src.data_utils.download_retrieval_datasets import get_fiqa_dataset, get_squad_dataset

class Corpus:
    def __init__(self, dataset_name: str, base_dir: str):
        self.dataset_name = dataset_name
        if not os.path.exists(f'{base_dir}/{self.dataset_name}'):
            raw_data_dir = f'{base_dir}/raw'
            os.makedirs(raw_data_dir, exist_ok=True)
            # load and process data
            self.ds = self.__load_data(base_dir=raw_data_dir)
            self.__rename_columns()
            if self.dataset_name == 'squad':
                self.__wrap_squad_contexts()
            self.relevant_ids = self.__assign_ids()
            os.makedirs(f'{base_dir}/{self.dataset_name}', exist_ok=True)
            DatasetDict(self.ds).save_to_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json', 'w') as f:
                json.dump(self.relevant_ids, f)
            
            print(f'Saved {self.dataset_name} dataset to path {base_dir}/{self.dataset_name}.')
            shutil.rmtree(f'{base_dir}/raw')
        else:
            self.ds = load_from_disk(f'{base_dir}/{self.dataset_name}')
            with open(f'{base_dir}/{self.dataset_name}/relevant_ids.json', 'r') as f:
                self.relevant_ids = json.load(f)


    @property
    def train(self):
        return self.ds.get('train', None)
    
    @property
    def val(self):
        return self.ds.get('validation', None)

    @property
    def test(self):
        return self.ds.get('test', None)

    @property
    def all_documents(self):
        train_text, train_ids = list(self.train['text']), list(self.train['document_id'])
        val_text, val_ids =     list(self.val['text']), list(self.val['document_id'])
        
        all_text = train_text + val_text
        all_ids = train_ids   + val_ids

        if self.test is not None:
            test_text, test_ids = list(self.test['text']), list(self.test['document_id'])
            all_text += test_text
            all_ids  += test_ids

        # Keep each unique document once (a passage can be relevant to several queries)
        unique_documents = {}
        for texts, ids in zip(all_text, all_ids):
            for text, _id in zip(texts, ids):
                unique_documents[_id] = text

        return Dataset.from_dict({'text': [[text] for text in unique_documents.values()],
                                  'document_id': [[_id] for _id in unique_documents.keys()]})


    def __load_data(self, base_dir) -> dict[Dataset]:
        assert self.dataset_name in ['squad', 'fiqa'], f'Dataset {self.dataset_name} not supported. Options are ["squad", "fiqa"]'
        
        if self.dataset_name == 'fiqa':
            ds = get_fiqa_dataset(base_dir)

        elif self.dataset_name == 'squad':
            ds = get_squad_dataset(base_dir)

        return {key: ds[key] for key in ds.keys()}

    def __rename_columns(self) -> dict[Dataset]:
        column_mapping = {'fiqa':  'ground_truths',
                          'squad': 'context'
                         }
        
        for key in self.ds:
            self.ds[key] = self.ds[key].rename_column(column_mapping[self.dataset_name], 'text')
        return self.ds

    def __wrap_squad_contexts(self):
        # SQuAD stores one context string per question; the rest of the code expects a list of passages
        for key in self.ds:
            self.ds[key] = self.ds[key].map(lambda row: {'text': [row['text']]})

    def __assign_ids(self):
        q_id_counter = 0
        text_to_doc_id = {}  # identical passages share one id
        relevant_document_ids = {}

        for key in self.ds:
            query_ids = []
            document_ids = []
            for documents in self.ds[key]['text']:
                query_ids.append(f'q{q_id_counter}')
                relevant_ids = []
                for doc in documents:
                    if doc not in text_to_doc_id:
                        text_to_doc_id[doc] = f'doc{len(text_to_doc_id)}'
                    relevant_ids.append(text_to_doc_id[doc])
                
                document_ids.append(relevant_ids)
                
                relevant_document_ids[f'q{q_id_counter}'] = relevant_ids
                q_id_counter += 1
                
            self.ds[key] = self.ds[key].add_column('query_id', query_ids)
            self.ds[key] = self.ds[key].add_column('document_id', document_ids)

        return relevant_document_ids