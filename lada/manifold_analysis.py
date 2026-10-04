#!/usr/bin/env python3
"""
manifold_analysis.py - 流形几何分析脚本
输入：特征numpy文件(.npy)
输出：流形几何指标（本征维度、曲率谱、各向异性指数、邻域图拓扑）

用法: python3 manifold_analysis.py --features_dir /path/to/features --output_dir ./results
"""
import os, sys, json, numpy as np, argparse, warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 尝试导入skdim
try:
    from skdim.id import TwoNN
    SKDIM_AVAILABLE = True
except ImportError:
    SKDIM_AVAILABLE = False
    print("  [WARN] scikit-dimension not installed. ID estimation will use PCA method.")


def estimate_intrinsic_dim(features, method='two_nn'):
    """估计本征维度（Intrinsic Dimension）"""
    # subsample for speed (max 5000 points)
    n = min(5000, len(features))
    idx = np.random.RandomState(42).choice(len(features), n, replace=False)
    subset = features[idx]
    
    if method == 'two_nn' and SKDIM_AVAILABLE:
        two_nn = TwoNN()
        id_ = two_nn.fit_transform(subset)
        return float(id_)
    else:
        # PCA比值法（简化的MLE-like估计）
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(subset)
        # 找到解释95%方差的主成分数
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        id_pca = int(np.searchsorted(cumsum, 0.95) + 1)
        # 最大曲率点（elbow method）
        diffs = np.diff(pca.explained_variance_ratio_)
        id_elbow = int(np.argmax(np.abs(np.diff(diffs))) + 1) if len(diffs) > 1 else id_pca
        return {
            'id_pca_95': id_pca,
            'id_elbow': max(1, id_elbow),
        }


def compute_curvature_spectrum(features, n_neighbors=15, n_samples=3000):
    """估计局部主曲率谱（通过邻域PCA）"""
    n = min(n_samples, len(features))
    idx = np.random.RandomState(42).choice(len(features), n, replace=False)
    subset = features[idx]
    
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(subset)
    distances, indices = nn.kneighbors(subset)
    
    curvatures = []
    explained_var = []
    
    for i in range(n):
        neighbors = subset[indices[i][1:]]  # 不包括自身
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        
        # 曲率 ≈ 第一奇异值与迹的比值
        local_var = s.sum()
        if local_var > 1e-10:
            # 谱分析
            var_ratio = s ** 2 / (s ** 2).sum()
            curv = s[0] / local_var if local_var > 0 else 0
            curvatures.append(curv)
            explained_var.append(var_ratio[:5].tolist())
    
    if len(curvatures) > 0:
        return {
            'mean_curvature': float(np.mean(curvatures)),
            'std_curvature': float(np.std(curvatures)),
            'median_curvature': float(np.median(curvatures)),
            'curvature_skew': float(np.mean((curvatures - np.mean(curvatures))**3) / (np.std(curvatures)**3 + 1e-10)),
        }
    return {'mean_curvature': 0, 'std_curvature': 0, 'median_curvature': 0, 'curvature_skew': 0}


def compute_anisotropy(features):
    """计算各向异性指数"""
    from sklearn.decomposition import PCA
    
    # 完整特征协方差的特征值谱
    pca = PCA()
    pca.fit(features)
    eigvals = pca.explained_variance_
    eigvals = eigvals[eigvals > 1e-10]
    
    if len(eigvals) < 5:
        return {'anisotropy': 1.0, 'eig_slope': 0}
    
    # 幂律拟合: log(eigvals) = slope * log(rank) + intercept
    ranks = np.arange(1, len(eigvals) + 1)
    log_ranks = np.log10(ranks)
    log_eigs = np.log10(eigvals)
    
    # 只用前50%的特征值做拟合（避免尾部噪声）
    half = len(log_eigs) // 2
    slope, intercept = np.polyfit(log_ranks[:half], log_eigs[:half], 1)
    
    # 各向异性指数 = -slope（越陡峭越各向异性）
    anisotropy = -slope
    
    # 特征值均匀性（归一化熵）
    eig_norm = eigvals / eigvals.sum()
    entropy = -np.sum(eig_norm * np.log(eig_norm + 1e-10))
    max_entropy = np.log(len(eig_norm))
    uniformity = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'anisotropy': float(anisotropy),
        'eig_slope': float(slope),
        'eig_uniformity': float(uniformity),
        'n_components_95': int(np.searchsorted(np.cumsum(eigvals/eigvals.sum()), 0.95) + 1),
        'eig_dim': len(eigvals),
    }


def compute_knn_graph(features, k=10, n_samples=5000):
    """kNN邻域图分析"""
    n = min(n_samples, len(features))
    idx = np.random.RandomState(42).choice(len(features), n, replace=False)
    subset = features[idx]
    
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=2)
    nn.fit(subset)
    distances, indices = nn.kneighbors(subset)
    
    # 平均邻域距离
    mean_dist = np.mean(distances[:, 1:])  # 不包括自身
    
    # 邻域密度估计（第k近邻距离）
    kth_dist = distances[:, -1]
    density = 1.0 / (kth_dist + 1e-10)
    
    # 局部邻域各向异性（第k近邻距离的变异系数）
    cv = np.std(kth_dist) / (np.mean(kth_dist) + 1e-10)
    
    return {
        'mean_knn_dist': float(mean_dist),
        'kth_dist_mean': float(np.mean(kth_dist)),
        'kth_dist_std': float(np.std(kth_dist)),
        'knn_density_mean': float(np.mean(density)),
        'knn_cv': float(cv),
    }


def analyze_features(features_path, output_dir):
    """分析单个特征文件"""
    base = os.path.splitext(os.path.basename(features_path))[0]
    # 去掉_features后缀
    ckpt_name = base.replace('_features', '')
    
    out_file = os.path.join(output_dir, f"{ckpt_name}_geometry.json")
    if os.path.exists(out_file):
        with open(out_file) as f:
            return json.load(f)
    
    print(f"\n📊 分析: {ckpt_name}")
    
    # 加载特征
    features = np.load(features_path)
    print(f"  特征形状: {features.shape}")
    
    if len(features) > 50000:
        # 随机采样50,000点
        idx = np.random.RandomState(42).choice(len(features), 50000, replace=False)
        features = features[idx]
        print(f"  采样至: {features.shape}")
    
    result = {'num_samples': len(features), 'feat_dim': features.shape[1]}
    
    # 1. 本征维度
    print("  [1/4] 本征维度估算...")
    result['intrinsic_dim'] = estimate_intrinsic_dim(features)
    
    # 2. 曲率谱
    print("  [2/4] 曲率谱分析...")
    result['curvature'] = compute_curvature_spectrum(features)
    
    # 3. 各向异性
    print("  [3/4] 各向异性分析...")
    result['anisotropy'] = compute_anisotropy(features)
    
    # 4. kNN邻域图
    print("  [4/4] 邻域图分析...")
    result['knn_graph'] = compute_knn_graph(features)
    
    # 保存
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"  结果保存至: {out_file}")
    return result


def summarize_results(output_dir):
    """汇总所有分析结果为表格（用于生成Table 1）"""
    results = []
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('_geometry.json'):
            with open(os.path.join(output_dir, f)) as fh:
                data = json.load(fh)
            ckpt = f.replace('_geometry.json', '')
            row = {
                'checkpoint': ckpt,
                'num_samples': data.get('num_samples', ''),
                'feat_dim': data.get('feat_dim', ''),
            }
            
            id_ = data.get('intrinsic_dim', {})
            if isinstance(id_, dict):
                row['id_pca_95'] = id_.get('id_pca_95', '')
                row['id_elbow'] = id_.get('id_elbow', '')
            else:
                row['id'] = id_
            
            curv = data.get('curvature', {})
            row['mean_curvature'] = round(curv.get('mean_curvature', 0), 6)
            row['median_curvature'] = round(curv.get('median_curvature', 0), 6)
            
            aniso = data.get('anisotropy', {})
            row['anisotropy'] = round(aniso.get('anisotropy', 0), 4)
            row['uniformity'] = round(aniso.get('eig_uniformity', 0), 4)
            row['n_components_95'] = aniso.get('n_components_95', '')
            
            knn = data.get('knn_graph', {})
            row['mean_knn_dist'] = round(knn.get('mean_knn_dist', 0), 4)
            row['knn_cv'] = round(knn.get('knn_cv', 0), 4)
            
            results.append(row)
    
    # 打印表格
    if results:
        print("\n" + "=" * 120)
        header = ['Checkpoint', 'ID', 'Curvature', 'Anisotropy', 'Uniformity', 'kNN dist']
        print(f"{'Checkpoint':<35} {'ID':<8} {'Curvature':<12} {'Anisotropy':<12} {'Uniformity':<12} {'kNN dist':<10}")
        print("-" * 120)
        for r in results:
            id_str = str(r.get('id', r.get('id_pca_95', '')))
            print(f"{r['checkpoint']:<35} {id_str:<8} {r['mean_curvature']:<12} {r['anisotropy']:<12} {r['uniformity']:<12} {r['mean_knn_dist']:<10}")
    
    # 保存汇总
    summary_path = os.path.join(output_dir, "manifold_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n汇总结果保存至: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, default='./features',
                        help='特征文件目录')
    parser.add_argument('--output_dir', type=str, default='./features/analysis',
                        help='分析结果输出目录')
    parser.add_argument('--all', action='store_true', help='分析所有特征文件')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                        help='具体特征文件')
    parser.add_argument('--summarize', action='store_true', help='汇总已有结果')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.summarize:
        summarize_results(args.output_dir)
        return
    
    # 确定要分析的文件
    if args.all:
        feature_files = sorted([
            os.path.join(args.features_dir, f)
            for f in os.listdir(args.features_dir)
            if f.endswith('_features.npy')
        ])
    elif args.features:
        feature_files = [os.path.join(args.features_dir, f) for f in args.features]
    else:
        # 默认分析所有
        feature_files = sorted([
            os.path.join(args.features_dir, f)
            for f in os.listdir(args.features_dir)
            if f.endswith('_features.npy')
        ])
    
    print(f"找到 {len(feature_files)} 个特征文件")
    
    for fpath in feature_files:
        try:
            analyze_features(fpath, args.output_dir)
        except Exception as e:
            print(f"  ❌ 分析失败 {fpath}: {e}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("  生成汇总表...")
    summarize_results(args.output_dir)
    print(f"\n完成！结果在: {args.output_dir}")


if __name__ == '__main__':
    main()
