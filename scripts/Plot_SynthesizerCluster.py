from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BaseDir='/pscratch/sd/n/nkyriaco/TauCP/TauRecoML/OmniLearned/plots/03_23_26/PerClusterFeatures/'
ProcessDir = BaseDir + 'Process/'
DecayModeDir = BaseDir + 'DecayMode/' 

ClusterFeatureNames = [ 'deta', 'dphi', 'center_lambda', 'center_mag', 'e_log', 'em_probability', 'et_log', 'eta', 'first_eng_dens', 'phi', 'second_lambda', 'second_r']
#ClusterFeatureNames = [ 'deta', 'dphi']  # Testing purposes
MaxClusters = 20


# Load in Track Kinematic Breakdown by DecayMode
with PdfPages('PerCluster_Kinematics_DecayModeBreakdown.pdf') as pdf:
    # Generate all image paths
    image_list = []
    for cls_idx in range(MaxClusters):  # Loop over the (pT-sorted) cluster indices
        for feature in ClusterFeatureNames:  # Loop over the features for each cluster
            img_path = f"{DecayModeDir}cluster_by_decay_mode_cluster_{cls_idx}_cls_{feature}.png"
            image_list.append((img_path, f"Cluster {cls_idx} - {feature}"))
    
    # Create pages with 4 images per page (2x2 grid)
    for page_idx in range(0, len(image_list), 4):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # Flatten to 1D for easier indexing
        
        # Add up to 4 images on this page
        for subplot_idx in range(4):
            ax = axes[subplot_idx]
            img_idx = page_idx + subplot_idx
            
            if img_idx < len(image_list):
                img_path, title = image_list[img_idx]
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.set_title(title, fontsize=11, fontweight='bold')
            
            ax.axis('off')
        
        fig.suptitle(f'Page {page_idx // 4 + 1}', fontsize=16)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

with PdfPages('PerCluster_Kinematics_ProcessBreakdown.pdf') as pdf:
    # Generate all image paths
    image_list = []
    for cls_idx in range(MaxClusters):  # Loop over the (pT-sorted) cluster indices
        for feature in ClusterFeatureNames:  # Loop over the features for each cluster
            img_path = f"{ProcessDir}cluster_cluster_{cls_idx}_cls_{feature}.png"
            image_list.append((img_path, f"Cluster {cls_idx} - {feature}"))
    
    # Create pages with 4 images per page (2x2 grid)
    for page_idx in range(0, len(image_list), 4):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # Flatten to 1D for easier indexing
        
        # Add up to 4 images on this page
        for subplot_idx in range(4):
            ax = axes[subplot_idx]
            img_idx = page_idx + subplot_idx
            
            if img_idx < len(image_list):
                img_path, title = image_list[img_idx]
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.set_title(title, fontsize=11, fontweight='bold')
            
            ax.axis('off')
        
        fig.suptitle(f'Page {page_idx // 4 + 1}', fontsize=16)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()