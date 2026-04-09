from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BaseDir='/pscratch/sd/n/nkyriaco/TauCP/TauRecoML/OmniLearned/plots/03_23_26/PerTrackFeatures/'
ProcessDir = BaseDir + 'Process/'
DecayModeDir = BaseDir + 'DecayMode/' 

TrackFeatureNames = [ 'pt_log', 'deta', 'dphi', 'e_log', 'z0', 'z0sintheta', 'd0', 'ntrthits', 'ntrthighthresholdhits', 'nscthits', 'npixelhits', 'nblayerhits']
MaxTracks = 15

#TrackFeatureNames = [ 'pt_log', 'deta', 'dphi', 'e_log'] # Testing purposes
#MaxTracks = 1 # Testing purposes

# Load in Track Kinematic Breakdown by DecayMode
with PdfPages('PerTrack_Kinematics_DecayModeBreakdown.pdf') as pdf:
    # Generate all image paths
    image_list = []
    for trk_idx in range(MaxTracks):  # Loop over the (pT-sorted) track indices
        for feature in TrackFeatureNames:  # Loop over the features for each track
            img_path = f"{DecayModeDir}track_by_decay_mode_track_{trk_idx}_trk_{feature}.png"
            image_list.append((img_path, f"Track {trk_idx} - {feature}"))
    
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

with PdfPages('PerTrack_Kinematics_ProcessBreakdown.pdf') as pdf:
    # Generate all image paths
    image_list = []
    for trk_idx in range(MaxTracks):  # Loop over the (pT-sorted) track indices
        for feature in TrackFeatureNames:  # Loop over the features for each track
            img_path = f"{ProcessDir}track_track_{trk_idx}_trk_{feature}.png"
            image_list.append((img_path, f"Track {trk_idx} - {feature}"))
    
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