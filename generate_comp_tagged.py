import save_test_embedding 
from LSTM_KNN import y

file_path = "data/test.untagged"
#GENERATING TAGGED TEST FILE AS "test.tagged"

# Open the file for writing
with open("test.tagged", "w", encoding="utf-8") as tagged_test:
    # tagged_test.truncate(0)
    # Copy from untagged
    with open("data/test.untagged", "r", encoding="utf-8") as untagged_test:
        lines = untagged_test.readlines()    
    tagged_test.truncate(0)
    # Write the first line (the visual studio text editor does not accept .tagged file that starts with &)
    tagged_test.write("&\tO\n")
    # Write predictions
    y_index = 1
    for i, line in enumerate(lines):
        if i==0:
            continue
        if line!="" and line!="\n" and line!="\t\n" and line!=" ":  # Check if line is not empty
            if y[y_index] == 0:
                tag = "O"
            else:
                tag = "I"
            lines[i] = line.rstrip() + "\t" + tag + "\n"
            y_index += 1

    # Write updated lines to the tagged file
    tagged_test.writelines(lines[1:])
print("Done tagging test.")
