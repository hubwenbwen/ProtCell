# tokenizer.py

# tokenizer.py

from typing import List, Dict, Union

class ProteinTokenizer:
    def __init__(self, protein_names: List[str], special_tokens: Dict[str, int] = None):
        """
        Args:
            protein_names: list of all protein symbols
            special_tokens: e.g., {"<pad>": 0, "<eos>": 1}
        """
        if special_tokens is None:
            special_tokens = {"<pad>": 0, "<eos>": 1}

        self.special_tokens = special_tokens
        self.pad_token = next(k for k, v in special_tokens.items() if k.lower() == "<pad>")
        self.eos_token = next(k for k, v in special_tokens.items() if k.lower() == "<eos>")
        self.pad_token_id = self.special_tokens[self.pad_token]
        self.eos_token_id = self.special_tokens[self.eos_token]

        # 保留顺序、去重，避免副作用
        protein_names = list(dict.fromkeys(protein_names))

        # 起始 index 取决于已有的 special token 个数
        offset = len(self.special_tokens)

        # 构建词表（special + protein names）
        self.protein_to_id = {**self.special_tokens}
        for i, name in enumerate(protein_names):
            self.protein_to_id[name] = i + offset

        self.id_to_protein = {idx: name for name, idx in self.protein_to_id.items()}
        self.vocab_size = len(self.protein_to_id)

    def encode(self, protein_input: Union[str, List[str]], add_eos: bool = False) -> List[int]:
        """
        Converts a list of protein names into token IDs.
        """
        if isinstance(protein_input, str):
            return self.protein_to_id[protein_input]
        else:
            return [self.protein_to_id[p] for p in protein_input]

    def decode(self, token_ids: Union[int, List[int]], remove_special: bool = False) -> List[str]:
        """
        Converts token IDs back to protein names.
        """
        if isinstance(token_ids, int):
            return self.id_to_protein[token_ids]
        names = [self.id_to_protein[i] for i in token_ids]
        if remove_special:
            names = [n for n in names if n not in self.special_tokens]
        return names

    def __len__(self):
        return self.vocab_size
    
    def get_token_weights(self, protein_weights: Dict[str, float], default_weight: float = 1.0) -> List[float]:
        """
        根据蛋白质出现频率创建与token ID对应的权重列表
        
        Args:
            tokenizer: ProteinTokenizer实例
            protein_weights: 蛋白质名称到权重的字典
            default_weight: 特殊token和未出现在protein_weights中的蛋白质的默认权重
        
        Returns:
            权重列表，索引对应token ID
        """
        weight_list = [default_weight] * self.vocab_size
        
        for protein, token_id in self.protein_to_id.items():
            if protein in protein_weights:
                weight_list[token_id] = protein_weights[protein]
            elif protein in self.special_tokens:
                weight_list[token_id] = default_weight
            else:
                weight_list[token_id] = default_weight
        
        return weight_list

    def add_tokens(self, new_proteins: List[str]):
        """将新的蛋白加入词表，返回新加入的token数"""
        added = 0
        for name in new_proteins:
            if name not in self.protein_to_id:
                idx = len(self.protein_to_id)
                self.protein_to_id[name] = idx
                self.id_to_protein[idx] = name
                added += 1
        self.vocab_size = len(self.protein_to_id)
        return added

    def save_tokenizer(self, filepath: str):
        """保存tokenizer到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "protein_to_id": self.protein_to_id,
                "special_tokens": self.special_tokens
            }, f)
    
    def get_all_proteins(self):
        """返回所有蛋白质名称（不包括特殊token）"""
        return [name for name in self.protein_to_id if name not in self.special_tokens]
            
    @classmethod
    def load_tokenizer(cls, filepath: str):
        """从文件加载tokenizer"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        tokenizer = cls(protein_names=[], special_tokens=data["special_tokens"])
        tokenizer.protein_to_id = data["protein_to_id"]
        tokenizer.id_to_protein = {idx: name for name, idx in tokenizer.protein_to_id.items()}
        tokenizer.vocab_size = len(tokenizer.protein_to_id)
        return tokenizer

if __name__ == "__main__":
    # protein_names = ["TP53", "EGFR", "CDK2"]
    # tokenizer = ProteinTokenizer(protein_names)

    # print(tokenizer.encode("TP53", add_eos=True))  # [2, 3, 1]
    # print(tokenizer.decode([0, 3, 1]))  # ['TP53', 'EGFR', '<eos>']
    
    # new_proteins = ["BRCA1", "MYC", "TP53"]
    # added_count = tokenizer.add_tokens(new_proteins)
    # print(tokenizer.encode("BRCA1"))  # New token ID
    # print(tokenizer.decode([0, 3, 1, 4, 5]))  # ['<pad>', 'EGFR', '<eos>', 'BRCA1', 'MYC']
    # tokenizer.save_tokenizer("tokenizer_test.json")
    # loaded_tokenizer = ProteinTokenizer.load_tokenizer("tokenizer_test.json")
    # print(loaded_tokenizer.decode([0, 3, 1, 4, 5]))  # ['<pad>', 'EGFR', '<eos>', 'BRCA1', 'MYC']
    
    import pandas as pd
    protein_count = pd.read_csv("/data/hubowen/protein_data/protein_count.csv", sep=",")
    protein_names = protein_count["var_name"].tolist()
    tokenizer = ProteinTokenizer(protein_names)
    tokenizer.save_tokenizer("tokenizer_all_vars.json")

