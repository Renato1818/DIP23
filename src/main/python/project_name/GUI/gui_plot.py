
import matplotlib.pyplot as plt
import numpy as np
import cv2

class GuiPlot:
    def __init__(self, figsize=(8, 5)):
        self.figsize=figsize
        return
    
    def find_scope_plot_bar(self, types_with_scores):
        type_names = types_with_scores[:, 0]
        num_images = types_with_scores[:, 1]
        avg_similarity = types_with_scores[:, 2] / np.maximum(1, num_images)  # Avoid division by zero
        
        # Create a figure with a specified size
    
        fig, ax = plt.subplots(figsize=self.figsize)
        ind = np.arange(len(type_names))

        bars = ax.bar(ind, avg_similarity, label='Average Similarity', color='tab:blue')

        ax.set_xlabel('Flower Types')
        ax.set_ylabel('Average Similarity')
        ax.set_title('Average Similarity by Flower Type')
        ax.set_xticks(ind)
        ax.set_xticklabels(type_names, rotation='vertical')  # Rotate labels vertically
        ax.legend()

        # Annotate bars with their corresponding values
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.02, f'{avg_similarity[i]:.2f}', ha='center')
        
        plt.show(block=False)
    
    def plot_images(self, new_image_path, most_similar_img, most_similar_result):
        # Plot the images
        fig, axs = plt.subplots(1, 2, figsize=(8, 5))
        axs[0].imshow(cv2.cvtColor(cv2.imread(new_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        axs[0].set_title("Image to classify")
        axs[0].axis("off")
        axs[1].imshow(cv2.cvtColor(most_similar_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Most Similar Image\nType: {most_similar_result.folder_name}\nScore: {most_similar_result.similarity_score}")
        axs[1].axis("off")
        plt.show()
    
