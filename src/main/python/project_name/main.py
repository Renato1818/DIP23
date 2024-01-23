import DataBase.database as db
from Compare import compare as cp
from Compare import sift
import sys

database_path="C:/Users/asus/GitHub_clones/DIP23/src/resources/data_base"
#In case, different path of database: 
# database_path = pt.GuiPlot.path_directory()
if database_path == -1:
    sys.exit()

# Initialize classes
database = db.Database(database_path)
image_comparer = cp.Compare(sift.Sift())

# Run the code
database.change_k(10)
database.change_test(1)
image_comparer.trainning(database)

