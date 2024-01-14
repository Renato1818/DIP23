import cv2
import numpy as np
from tqdm import tqdm
import numpy as np
import cv2
import SVM.sift as sift
import DataBase.database as db
import Terminal.terminal as term
import GUI.gui_plot as pt
import os


class ResultStructure:
    def __init__(self, image_name, folder_name, similarity_score, database_path):
        self.image_name = image_name
        self.folder_name = folder_name
        self.similarity_score = similarity_score
        self.database_path = database_path


class Compare:
    def __init__(self, sift_comparer, terminal=False, display=True):
        self.sift_comparer = sift_comparer
        self.terminal = terminal
        self.display = display
        self.plot = pt.GuiPlot()
        self.term = term.Terminal()

    def compare_vector(self, database: db, image_test_vector):
        # Initialize an empty list to store results for each image
        all_results = []

        for image_test in image_test_vector:
            # Compare the current image with images from the database
            results = self.compare(database, image_test)
            if results != -1:
                all_results.append(results)

        if not all_results:
            return -1

        return all_results

    def compare(self, database: db, image_test):
        # Read image paths from the database
        types, database_image_paths = database.read_images()
        if not database_image_paths:
            return -1

        # Compare the test image with images from the database
        results = self.compare_with_database(image_test, database_image_paths)

        ## Statistics ##
        # Scope for each type of flowers
        types_with_scores = self.find_scope(types, results)
        # Display the most similar image
        most_similar_result = max(results, key=lambda x: x.similarity_score)
        self.term.similar_image(types_with_scores, most_similar_result)

        most_similar_img = cv2.imread(
            most_similar_result.database_path, cv2.IMREAD_COLOR
        )
        # Plot the images
        if self.display:
            self.plot.plot_images(image_test, most_similar_img, most_similar_result)

        # sift.compare_images_sift(image_test, most_similar_result.database_path)

        return results

    def compare_with_database(self, new_image_path, database_image_paths):
        results = []

        # Load the new image
        new_img = cv2.imread(new_image_path, cv2.IMREAD_GRAYSCALE)

        for database_image_path, folder_name in tqdm(
            database_image_paths, desc="Comparing images", unit="image"
        ):
            # Load the database image
            database_img = cv2.imread(database_image_path, cv2.IMREAD_GRAYSCALE)
            # print(f"Image: {database_image_path}; Is path: {folder_name}")

            # Perform the comparison using the SIFT comparer
            similarity_score = self.sift_comparer.compare_images(new_img, database_img)

            # Append the result to the list
            result = ResultStructure(
                image_name=os.path.basename(new_image_path),
                folder_name=folder_name,
                similarity_score=similarity_score,
                database_path=database_image_path,
            )
            results.append(result)

            if self.terminal:
                print(
                    f"Image: {result.image_name}, Folder: {result.folder_name}, Score: {result.similarity_score}"
                )

        return results

    def find_scope(self, types, results: list[ResultStructure]):
        types_array = np.array(types)
        types_with_scores = np.zeros((len(types), 3), dtype=object)

        for result in results:
            indices = np.where(types_array == result.folder_name)
            for i in indices[0]:
                types_with_scores[i, 0] = result.folder_name
                types_with_scores[i, 1] += 1  # number images
                types_with_scores[i, 2] += result.similarity_score  # total scope

        # Filter out scores with 0 images
        types_with_scores = types_with_scores[types_with_scores[:, 1] > 0]

        # Print a bar plot
        if self.display:
            self.plot.find_scope_plot_bar(types_with_scores)

        # Order types_with_scores by average similarity score in descending order
        order = np.argsort(
            types_with_scores[:, 2] / np.maximum(1, types_with_scores[:, 1])
        )[::-1]
        types_with_scores = types_with_scores[order]

        # return types_with_scores.tolist()
        return types_with_scores
