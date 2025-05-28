import json, copy
import numpy as np
import os

def base_structure() -> dict:
    return {"X" : [], "y" : []}


def load_data(file_name : str) -> dict:
    data_loaded = {}
    try:
        with open(f"{file_name}.json", "r") as data_file:
            data_loaded = json.load(data_file)
    except FileNotFoundError as error:
        print("The file was not found: ", error)
        return None
    return data_loaded


def reset_data_to_base(file_name : str) -> None:
    with open(f"{file_name}.json", "w") as arq:
        json.dump(base_structure(), arq, indent=5)


def add_data(file_name : str, data_to_add : dict):
    """Add the new data to existing data in file while removing all duplicates"""
    if len(data_to_add["X"]) != len(data_to_add["y"]):
        raise(ValueError("X length has to be equals to ys"))
    
    if not os.path.exists(f"{file_name}.json"):
        with open(f"{file_name}.json", "w") as arq:
            json.dump(base_structure(), arq, indent=5)


    with open(f"{file_name}.json", "r") as data_file:
        try:
            data_loaded : dict = json.load(data_file)
        except json.JSONDecodeError as err:
            print(f"{err}, going to create new data")
            data_loaded = base_structure()
        new_data : dict = base_structure()

        keys = data_loaded.keys()

        items_already_existent : list = []
        indexes_to_remove : list = []
        
        for _ in range(2):
            for key in keys:
                if _ == 0:
                    values_to_add : list = copy.deepcopy(data_to_add[key])
                    try:
                        values_existent : list = copy.deepcopy(data_loaded[key])
                    except KeyError as error:
                        values_existent : list = []
                    
                    new_list : list = values_to_add + values_existent
                    new_data[key] = copy.deepcopy(new_list)

                    if key == "X":
                        for index in range(len(new_list)):
                            if new_list[index] not in items_already_existent:
                                items_already_existent.append(new_list[index])
                                continue
                            indexes_to_remove.append(index)
                    continue

                items_to_remove = [copy.deepcopy(new_data[key][index]) for index in indexes_to_remove]
                for item in items_to_remove:
                    new_data[key].remove(item)

    with open(f"{file_name}.json", "w") as data_file:   
        json.dump(new_data, data_file, indent=5 )


def files_merger(file1_name : str, file2_name : str, new_file_name : str):
    json1_file_name = f"{file1_name}.json"
    json2_file_name = f"{file2_name}.json"
    if not (os.path.exists(json1_file_name) and os.path.exists(json2_file_name)):
        f1_exists = True
        f2_exists = True
        
        if not (os.path.exists(json1_file_name) or os.path.exists(json2_file_name)):
            f1_exists = False
            f2_exists = False
        elif not os.path.exists(json2_file_name):
            f2_exists = False
        else:
            f1_exists = False

        raise FileNotFoundError(f"Both files to be merged need to exist: file1: {f1_exists}, file2: {f2_exists}")

    data1 = load_data(file1_name)
    data2 = load_data(file2_name)
    add_data(new_file_name, data1)
    add_data(new_file_name, data2)



def f(x):
    return x*2
    


if __name__ == "__main__":
    
    
    data_add = {
        "X" : [i for i in range(100, 200)],
        "y" : [f(i) for i in range(100, 200)]
    }
    add_data("test_file_1", data_add)

    data_add = {
        "X" : [i for i in range(200, 400)],
        "y" : [f(i) for i in range(200, 400)]
    }
    add_data("test_file_2", data_add)

    files_merger("test_file_1", "test_file_2", "merged_file")
    

