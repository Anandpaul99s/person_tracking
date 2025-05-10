import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict


class CrossVideoMatcher:
    def __init__(self, similarity_threshold=0.6):
        """
        Initialize the cross-video matcher.

        Args:
            similarity_threshold: Cosine similarity threshold to consider two embeddings a match
        """
        self.similarity_threshold = similarity_threshold
        self.next_global_id = 1
        self.id_mapping = {}  # Map (video, track_id) to global ID

    def match_tracks(self, all_tracks):
        """
        Match tracks across videos and assign global person IDs.

        Args:
            all_tracks: List of dicts with keys: 'track_id', 'video', 'frame', 'bbox', 'embedding'

        Returns:
            List of dicts with keys: 'final_id', 'video', 'frame', 'bbox'
        """
        print(f"Starting cross-video matching with {len(all_tracks)} tracks")

        # Step 1: Group embeddings per unique (video, track_id)
        grouped_tracks = defaultdict(list)
        for track in all_tracks:
            key = (track['video'], track['track_id'])
            grouped_tracks[key].append(track['embedding'])

        print(f"Grouped into {len(grouped_tracks)} unique track identities")

        # Step 2: Compute average embedding per unique track
        representatives = {}
        for key, emb_list in grouped_tracks.items():
            avg = np.mean(emb_list, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm  # Normalize
            representatives[key] = avg

        print(f"Generated {len(representatives)} representative embeddings")

        # Step 3: Cluster based on cosine similarity
        assigned = set()
        clusters = []
        sorted_keys = sorted(representatives.keys())

        for key in sorted_keys:
            if key in assigned:
                continue

            base_emb = representatives[key]
            cluster = [key]
            assigned.add(key)

            for other_key in sorted_keys:
                if other_key in assigned or other_key == key:
                    continue
                other_emb = representatives[other_key]
                similarity = 1 - cosine(base_emb, other_emb)
                if similarity >= self.similarity_threshold:
                    cluster.append(other_key)
                    assigned.add(other_key)

            clusters.append(cluster)

        print(f"Formed {len(clusters)} clusters")

        # Step 4: Assign global IDs
        for cluster in clusters:
            gid = self.next_global_id
            self.next_global_id += 1
            for key in cluster:
                self.id_mapping[key] = gid

        # Step 5: Map global IDs to original tracks
        final_results = []
        for track in all_tracks:
            key = (track['video'], track['track_id'])
            global_id = self.id_mapping.get(key, -1)
            final_results.append({
                'final_id': global_id,
                'video': track['video'],
                'frame': track['frame'],
                'bbox': track['bbox']
            })

        print(
            f"Assigned {len(set(r['final_id'] for r in final_results))} unique global IDs")
        return final_results
