import os
import pandas as pd
from ec_number_prediction.data_processing_pipeline._utils import scale_up_cd_hit


class IdentityCluster:

    def __init__(self, cluster, members, identity_threshold) -> None:
        
        self._cluster = cluster
        self._members = members
        self._identity_threshold = identity_threshold

    @property
    def cluster(self):
        return self._cluster
    
    @property
    def members(self):
        return self._members
    
    @property
    def identity_threshold(self):
        return self._identity_threshold
    
    def __len__(self):
        return len(self._members)
    
    def add(self, member):
        self._members.append(member)

    def member_to_other_members(self, member):
        members_copy = self._members.copy()
        members_copy.remove(member)
        return members_copy

    

class ClustersIdentifier:

    def __init__(self, identity_threshold, cluster_to_members, member_to_cluster) -> None:
        self._identity_threshold = identity_threshold
        self._cluster_to_members = cluster_to_members
        self._member_to_cluster = member_to_cluster

    @property
    def identity_threshold(self):
        return self._identity_threshold
    
    @property
    def cluster_to_members(self):
        return self._cluster_to_members
    
    @cluster_to_members.setter
    def cluster_to_members(self, value):
        self._cluster_to_members = value

    @property
    def member_to_cluster(self):
        return self._member_to_cluster
    
    @member_to_cluster.setter
    def member_to_cluster(self, value):
        self._member_to_cluster = value

    def get_cluster_by_member(self, member):
        try:
            return self._cluster_to_members[self._member_to_cluster[member]]
        except KeyError:
            return None
    
    def __getitem__(self, key):
        return self._cluster_to_members[key]

    @classmethod
    def from_files(cls, folder, filename, identity_threshold):
        '''
        Go through the cluster files and collect
        all the cluster members, while indicating
        which belongs where.
        '''
        # get a list of filenames
        CLUSTER_FILES = [
            os.path.join(folder, filename + f"_clustered_sequences_{sim}.fasta.clstr")
            for sim in range(100, identity_threshold - 1, -10)
        ]

        # collect all cluster members
        clusters = scale_up_cd_hit(CLUSTER_FILES)
        ind_clusters = {}
        i = 0
        for clus in clusters:
            ind_clusters[i] = [clus] + clusters[clus]
            i += 1

        # convert to format that is suitable for data frames
        # clusters_for_df = {'cluster': [], 'member': []}
        cluster_to_members = {}
        member_to_cluster = {}
        for ind in ind_clusters:
            for member in ind_clusters[ind]:
                # clusters_for_df['cluster'].append(ind)
                # clusters_for_df['member'].append(member)
                if ind not in cluster_to_members:
                    cluster_to_members[ind] = IdentityCluster(ind, [], identity_threshold)
                
                cluster_to_members[ind].add(member)
                member_to_cluster[member] = ind

        # df = pd.DataFrame.from_dict(clusters_for_df, orient='columns')

        return cls(identity_threshold, cluster_to_members, member_to_cluster)