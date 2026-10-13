import numpy as np

def silhouette_score(X, labels):
    """
    Menghitung koefisien Silhouette rata-rata dari semua sampel.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Matriks fitur input.
    labels : array-like of shape (n_samples,)
        Label cluster untuk setiap sampel.
        
    Returns
    -------
    score : float
        Rata-rata koefisien Silhouette.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1 or n_clusters >= n_samples:
        return 0.0
    
    # Hitung matriks jarak berpasangan (Euclidean)
    # dists[i, j] = sqrt(max(||X[i] - X[j]||^2, 0))
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(X_sq + X_sq.T - 2 * np.dot(X, X.T), 0.0))
    
    silhouettes = np.zeros(n_samples)
    
    for i in range(n_samples):
        c_i = labels[i]
        
        # Mask untuk sampel lain di cluster yang sama
        same_cluster_mask = (labels == c_i)
        same_cluster_mask[i] = False # abaikan diri sendiri
        
        n_same = np.sum(same_cluster_mask)
        if n_same == 0:
            silhouettes[i] = 0.0
            continue
            
        a_i = np.mean(dists[i, same_cluster_mask])
        
        # Cari rata-rata jarak terdekat ke cluster lain
        b_i = float('inf')
        for c in unique_labels:
            if c == c_i:
                continue
            other_cluster_mask = (labels == c)
            if np.any(other_cluster_mask):
                mean_dist = np.mean(dists[i, other_cluster_mask])
                if mean_dist < b_i:
                    b_i = mean_dist
                    
        silhouettes[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
        
    return np.mean(silhouettes)


def _kmeans_plusplus(X, n_clusters, random_state):
    """
    Inisialisasi centroid menggunakan algoritma k-means++ dengan uji coba lokal
    agar serupa dengan perilaku scikit-learn.
    """
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    indices = np.full(n_clusters, -1, dtype=int)
    
    # Jumlah uji coba lokal
    n_local_trials = 2 + int(np.log(n_clusters))
    sample_weight = np.ones(n_samples, dtype=X.dtype)
    
    # Pilih center pertama secara acak
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    centers[0] = X[center_id]
    indices[0] = center_id
    
    closest_dist_sq = np.sum((X - centers[0])**2, axis=1)
    current_pot = np.sum(closest_dist_sq * sample_weight)
    
    for c in range(1, n_clusters):
        # Sampling kandidat center secara proporsional dengan closest_dist_sq
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(sample_weight * closest_dist_sq), rand_vals)
        candidate_ids = np.clip(candidate_ids, 0, closest_dist_sq.size - 1)
        
        # Hitung jarak dari semua kandidat ke semua titik
        distance_to_candidates = np.sum((X[candidate_ids][:, np.newaxis, :] - X[np.newaxis, :, :])**2, axis=2)
        
        # Hitung potensial kandidat
        temp_dists = np.minimum(closest_dist_sq, distance_to_candidates)
        candidates_pot = np.sum(temp_dists * sample_weight, axis=1)
        
        # Pilih kandidat terbaik yang meminimalkan potensial (inertia)
        best_candidate_idx = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate_idx]
        closest_dist_sq = temp_dists[best_candidate_idx]
        best_candidate = candidate_ids[best_candidate_idx]
        
        centers[c] = X[best_candidate]
        indices[c] = best_candidate
        
    return centers, indices


class KMeans:
    """
    Kelas KMeans kustom yang mensimulasikan perilaku scikit-learn.
    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.centroid_history_ = None  # list of centroid arrays per iterasi (best run)
        self.label_history_ = None    # list of label arrays per iterasi (best run)
        self.n_iter_ = 0               # jumlah iterasi hingga konvergen (best run)

    def fit(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} harus lebih besar atau sama dengan n_clusters={self.n_clusters}")
            
        master_rng = np.random.RandomState(self.random_state)
        seeds = master_rng.randint(np.iinfo(np.int32).max, size=self.n_init)
        
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        best_history = None
        best_label_history = None
        best_n_iter = 0
        
        for init_idx in range(self.n_init):
            run_rng = np.random.RandomState(seeds[init_idx])
            
            # Inisialisasi centroid
            if self.init == 'k-means++':
                centers, _ = _kmeans_plusplus(X, self.n_clusters, run_rng)
            else:
                indices = run_rng.choice(n_samples, self.n_clusters, replace=False)
                centers = X[indices].copy()

            # Rekam centroid awal (iterasi 0 = inisialisasi)
            # Untuk iter-0 belum ada assignment, gunakan label kosong (None)
            history = [centers.copy()]
            label_hist = [None]  # iterasi 0 belum ada assignment
            converged_iter = 0
                
            for iter_idx in range(self.max_iter):
                # Hitung jarak kuadrat Euclidean dari titik ke centroid
                X_sq = np.sum(X**2, axis=1, keepdims=True)
                C_sq = np.sum(centers**2, axis=1, keepdims=True).T
                dists_sq = np.maximum(X_sq + C_sq - 2 * np.dot(X, centers.T), 0.0)
                
                # Tetapkan label cluster terdekat
                labels = np.argmin(dists_sq, axis=1)
                
                # Perbarui centroid
                new_centers = np.empty_like(centers)
                for c in range(self.n_clusters):
                    mask = (labels == c)
                    if np.any(mask):
                        new_centers[c] = np.mean(X[mask], axis=0)
                    else:
                        # Tangani cluster kosong: pilih titik terjauh dari pusat terdekat
                        dists_to_assigned = np.sum((X - centers[labels])**2, axis=1)
                        furthest_idx = np.argmax(dists_to_assigned)
                        new_centers[c] = X[furthest_idx]
                        labels[furthest_idx] = c
                        
                # Cek konvergensi
                center_shift = np.sum((centers - new_centers)**2)
                centers = new_centers
                converged_iter = iter_idx + 1

                # Rekam centroid dan label setelah update
                history.append(centers.copy())
                label_hist.append(labels.copy())
                
                if center_shift < self.tol:
                    break
                    
            # Hitung inertia akhir untuk inisialisasi ini
            X_sq = np.sum(X**2, axis=1, keepdims=True)
            C_sq = np.sum(centers**2, axis=1, keepdims=True).T
            dists_sq = np.maximum(X_sq + C_sq - 2 * np.dot(X, centers.T), 0.0)
            labels = np.argmin(dists_sq, axis=1)
            inertia = np.sum(np.min(dists_sq, axis=1))
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_history = history
                best_label_history = label_hist
                best_n_iter = converged_iter
                
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.centroid_history_ = best_history  # list[ndarray(n_clusters, n_features)]
        self.label_history_ = best_label_history  # list[ndarray(n_samples,) | None]
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
