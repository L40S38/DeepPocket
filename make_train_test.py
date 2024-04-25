import os
from Bio import SeqIO
from Bio import Align
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import DBSCAN
from sklearn.model_selection import KFold
import numpy as np

# データディレクトリのパス
data_dir_A = './data_dir/scPDB_processed/'
data_dir_B = './data_dir/p2rank-datasets/holo4k/'

# .pdbファイルのパスを取得する関数
def get_pdb_files(directory):
    pdb_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(root, file))
    return pdb_files

# PDBファイルからアミノ酸配列を取得する関数
def get_sequence_from_pdb(pdb_file):
    # PDBParserのインスタンスを作成
    parser = PDBParser()

    # .pdbファイルをパースして構造を取得
    structure = parser.get_structure('structure', pdb_file)

    # アミノ酸配列を格納するリスト
    sequence = []

    # 構造からチェーンとアミノ酸を取得
    for model in structure:
        for chain in model:
            for residue in chain:
                # 3文字のアミノ酸コードを1文字のコードに変換
                sequence.append(seq1(residue.get_resname()))

    # リストを文字列に変換してアミノ酸配列を返す
    return ''.join(sequence)

# 配列相同性の計算
def compute_similarity(seq1, seq2):
    aligner = Align.PairwiseAligner()

    # アラインメントパラメータを設定
    aligner.match_score = 2
    aligner.mismatch_score = -1

    aligner.mode = "global" #ローカルの場合は"local", グローバルの場合は"global"
    print(f"seq1:{seq1}")
    print(f"seq2:{seq2}")
    alignments = aligner.align(seq1, seq2)
    alignment = alignments[0]

    length = alignment.shape[1]

    difference = np.array(alignment.aligned)[0][:, 1] - np.array(alignment.aligned)[0][:, 0]
    identity = np.sum(difference)
    ratio = identity / length
    print(f"ratio:{ratio}")
    return ratio

# rdkitのtanimoto類似度を計算する関数
def compute_tanimoto_similarity(mol1, mol2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# メイン処理
def main():
    # データセットAとBから.pdbファイルのパスを取得
    pdb_files_A = get_pdb_files(data_dir_A)
    pdb_files_B = get_pdb_files(data_dir_B)

    print(pdb_files_A,pdb_files_B)
    
    # 条件を満たす.pdbファイルを格納するリスト
    valid_pdb_files = []
    
    for pdb_A in pdb_files_A:
        seq_A = get_sequence_from_pdb(pdb_A)
        valid_flag = True
        for pdb_B in pdb_files_B:
            seq_B = get_sequence_from_pdb(pdb_B)
            
            # 条件①: 配列相同性が50%以上保持されていない
            similarity = compute_similarity(seq_A, seq_B)
            if similarity >= 0.5:
                valid_flag = False
                break 
            # 配列相同性が30%以上保持かつ結合リガンドの類似性が0.9以上ではない
            if similarity >= 0.3:
                mol_A = Chem.MolFromPDBFile(pdb_A)
                mol_B = Chem.MolFromPDBFile(pdb_B)
                
                if mol_A and mol_B:
                    tanimoto_similarity = compute_tanimoto_similarity(mol_A, mol_B)
                    
                    if tanimoto_similarity >= 0.9:
                        valid_flag = False
                        break                        
        
        if valid_flag:
            valid_pdb_files.append(pdb_A)
        
    print(f"num of valid files:{valid_pdb_files}")

    # 配列相同性30%でクラスタリング
    sequences = [get_sequence_from_pdb(pdb) for pdb in valid_pdb_files]
    X = [[compute_similarity(seq1, seq2) for seq1 in sequences] for seq2 in sequences]
    print(sequences,X)

    clustering = DBSCAN(eps=0.3, min_samples=5)
    clusters = clustering.fit_predict(X)
    
    # 10-fold cross-validationのためのデータを作成
    kf = KFold(n_splits=10)
    
    for i, (train_index, test_index) in enumerate(kf.split(valid_pdb_files)):
        train_files = [valid_pdb_files[idx][0] for idx in train_index]
        test_files = [valid_pdb_files[idx][0] for idx in test_index]
        
        with open(f'train{i}.txt', 'w') as f:
            f.write('\n'.join(train_files))
        
        with open(f'test{i}.txt', 'w') as f:
            f.write('\n'.join(test_files))

if __name__ == "__main__":
    main()
