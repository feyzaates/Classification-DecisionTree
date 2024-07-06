import csv
import math
import random
table = []
ES = 0
gain = []
varietyOfAtt = []
good = 0
bad = 0
index = -1
boundry = 0
exit = True

trainTable = []
testTable =[]
AccuracyRate = 0
report = []
class DecisionTree:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.children = []
        self.varrAtt = None
        self.es = None
        self.att = None

def fileReader(fileName):
    # reading from file for trainSet and testSet
    global table
    table = []
    with open(fileName, "r") as file:
        read = csv.reader(file)
        next(read)
        index=0
        for line in read:
            index+= 1
            table.insert(index, [line[0],float(line[1]),line[2],line[3],line[4],line[5],int(line[6]),line[7],line[8],line[9]])
            if index == 491:
                break

def calculateEntropy(node):
    # calculating ES for evey nodes
    global ES, good, bad
    ES = 0
    good = 0
    bad = 0
    for line in node.data:
        if line[len(node.data[0])-1] == "good":
            good += 1
        else:
            bad += 1

    numberOfElement = len(node.data)
    ES = (- ((good / numberOfElement) * math.log2(good / numberOfElement))
          - ((bad / numberOfElement) * math.log2(bad / numberOfElement)))


def calculations(updatedTable): # calculating gain for every node to decide children
    global ES, gain, varietyOfAtt
    gain = []
    varietyOfAtt = []
    for i in range(len(updatedTable[0])-1):
        countAttributes = {}
        countClass = {}
        calculateP = []
        calculateH = []
        attributeList = []
        global boundry
        if isinstance(updatedTable[0][i], (float, int)): # if part is for numerical attributes
            numbers = []
            countLessThan = 0
            countGreaterThan = 0
            for line in updatedTable:
                numbers.append(line[i])
            boundry = min(numbers) + ((max(numbers) - min(numbers)) / 2)
            # it is separate less and grater with using string counter
            for line in updatedTable:
                if line[i] <= boundry:
                    countLessThan += 1
                    classType = "less" + line[len(updatedTable[0]) - 1]
                    if classType in countClass:
                        countClass[classType] += 1
                    else:
                        countClass[classType] = 1
                elif line[i] > boundry:
                    countGreaterThan += 1
                    classType = "greater" + line[len(updatedTable[0]) - 1]
                    if classType in countClass:
                        countClass[classType] += 1
                    else:
                        countClass[classType] = 1
            # measurement for P and H
            calculateP.append(countLessThan / len(updatedTable))
            calculateP.append(countGreaterThan / len(updatedTable))
            attributeList.append("less")
            attributeList.append("greater")
            attributeList.append(float(boundry))
            if "lessgood" in countClass and "lessbad" in countClass:
                calculateH.append(
                    (- ((countClass["lessgood"] / countLessThan) * math.log2(countClass["lessgood"] / countLessThan))
                     - ((countClass["lessbad"] / countLessThan) * math.log2(countClass["lessbad"] / countLessThan))))
            else:
                calculateH.append(0)

            if "greatergood" in countClass and "greaterbad" in countClass:
                calculateH.append((- ((countClass["greatergood"] / countGreaterThan) * math.log2(
                    countClass["greatergood"] / countGreaterThan))
                                   - ((countClass["greaterbad"] / countGreaterThan) * math.log2(
                            countClass["greaterbad"] / countGreaterThan))))
            else:
                calculateH.append(0)

        else: # else part is for nominal attributes
            # this part count the variety of any attribute for example a and b for A1
            # with using string counter
            for line in updatedTable:
                attribute = line[i]
                if attribute in countAttributes:
                    countAttributes[attribute] += 1
                else:
                    attributeList.append(str(attribute))
                    countAttributes[attribute] = 1

                classType = attribute + line[len(updatedTable[0])-1]
                if classType in countClass:
                    countClass[classType] += 1
                else:
                    countClass[classType] = 1
            for string, numOfAtt in countAttributes.items():
                calculateP.append(numOfAtt / len(updatedTable))

                if string + "good" in countClass and string + "bad" in countClass:
                    calculateH.append((- ((countClass[string + "good"] / numOfAtt) * math.log2(countClass[string + "good"] / numOfAtt))
                                       - ((countClass[string + "bad"] / numOfAtt) * math.log2(countClass[string + "bad"] / numOfAtt))))
                else:
                    calculateH.append(0)
        # it shows every attribute and their variety in table
        varietyOfAtt.append(list(attributeList))
        HS = 0
        # calculating gain for every attribute in table
        for j in range(len(calculateP)):
            HS += calculateP[j] * calculateH[j]
        gain.append(ES - HS)

    #print(gain)



def buildNode(node): # it splits children with using gain list that calculated in calculations function
    calculateEntropy(node)   # calculating ES for evey nodes
    calculations(node.data)  # calculating gain for evey nodes
    node.varrAtt = varietyOfAtt  # and their variety of their att
    global index, exit
    exit = True
    if len(varietyOfAtt) == 0:   # when we do not need to split it exit from here
        exit = False             # for example if there is one kind of att
        return                   # it directly exits without doing any split

    maxEl = max(gain)
    index = gain.index(maxEl)  # index represents the attribute that has max gain
    #print(index, node.varrAtt)

    if len(node.varrAtt[index]) == 1 or (len(node.varrAtt[index]) == 3 and node.varrAtt[index][2] == 0):
        exit = False      # this if blok checks the comparator for less and greater cases
        return            # it could be 0 for some cases so we do not need to split
    if isinstance(node.data[0][index], (float, int)):   # for numeric attributes
        subSet1 = []
        subSet2 = []
        for line in node.data:
            if float(line[index]) <= node.varrAtt[index][2]:
                subSet1.append(list(line))
            else:
                subSet2.append(list(line))
        # split into two as less and greater
        # if everything same in class attribute the set as a leaf node
        allSame = all(row[len(subSet1[0]) - 1] == subSet1[0][len(subSet1[0]) - 1] for row in subSet1)
        if allSame:
            child = DecisionTree(subSet1[0][len(subSet1[0]) - 1], node.varrAtt[index][0])
        else:
            child = DecisionTree(subSet1, node.varrAtt[index][0])
        node.children.append(child)
        allSame = all(row[len(subSet2[0]) - 1] == subSet2[0][len(subSet2[0]) - 1] for row in subSet2)
        if allSame:
            child = DecisionTree(subSet2[0][len(subSet2[0]) - 1], node.varrAtt[index][1])
        else:
            child = DecisionTree(subSet2, node.varrAtt[index][1])
        node.children.append(child)
    else:   # for nominal attributes
        # split into number of variety of attribute
        for j in range(len(node.varrAtt[index])):
            subSet = []
            for line in node.data:
                if node.varrAtt[index][j] == line[index]:
                    subSet.append(list(line))
            # if everything same in class attribute the set as a leaf node
            allSame = all(row[len(subSet[0]) - 1] == subSet[0][len(subSet[0]) - 1] for row in subSet)
            if allSame:
                # if is a leaf node the data of node is good or bad
                child = DecisionTree(subSet[0][len(subSet[0]) - 1], node.varrAtt[index][j])
            else:
                # if is not a leaf node the data of node is a sub table of attribute
                child = DecisionTree(subSet, node.varrAtt[index][j])
            node.children.append(child)

    #print("parent ",node.name)
    #print(node.data)
    # if it is not leaf node
    # I prefer to delete the columns that related to children attribute
    # I added this part because when we calculate gain we do not
    # need to calculate grandparents attributes
    for k in range(len(node.children)):
        node.children[k].att = index
        if node.children[k].data != "good" and node.children[k].data != "bad":
            for i in range(len(node.children[k].data)):
                node.children[k].data[i].pop(index)
        #print(node.children[k].name, node.children[k].data)
    #print("\n")

def buildDecisionTree(node):
    # it is a recursive structure to build a decision tree
    for child in node.children:
        if child.data != "good" and child.data != "bad":
            # if the child of a node is not a leaf node then continue to build recursively
            buildNode(child)
            if exit:  # for hesitant nodes do not continue to build just stop in this step
                buildDecisionTree(child)

def calculateAccuracy(root,table):
    # traveling in decision tree with given table like train or test
    # to reach a leaf node or hesitant node
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    accuracy = 0
    global AccuracyRate
    for row in table:   # it checks for every line in data set
        i = 0
        node = root
        while i in range(len(node.children)):
            if node.children[i].name == "less" or node.children[i].name == "greater":
                # if the next node is numeric check the attribute in line index and
                # if it can find continue with related node
                index = node.children[i].att
                rowUpdate = list(row)
                rowUpdate.pop(index)
                if node.children[0].data != "good" and node.children[0].data != "bad":
                    for line in node.children[0].data:
                        if rowUpdate == line:
                            del row[index]
                            node = node.children[0]
                            i = -1
                            break  # when find in less node we do not check more
                elif node.children[1].data != "good" and node.children[1].data != "bad":
                    for line in node.children[1].data:
                        if rowUpdate == line:
                            del row[index]
                            node = node.children[1]
                            i = -1
                            break  # when find in greater we do not check more
                else:
                    if node.children[0].data == "good" or node.children[0].data == "bad":
                        del row[index]
                        node = node.children[0]
                        i = -1
                    elif node.children[1].data == "good" or node.children[1].data == "bad":
                        del row[index]
                        node = node.children[1]
                        i = -1
            else:
                # if the next node is nominal check the attribute in line index and
                # if it can find continue with related node
                if node.children[i].name == row[node.children[i].att]:
                    del row[node.children[i].att]
                    node = node.children[i]

                    i = -1
            i += 1
        if isinstance(node.data, str):
            if node.data == "good":
                if node.data == row[-1]:
                    TP += 1
                else:
                    FN += 1
            else:
                if node.data == row[-1]:
                    TN += 1
                else:
                    FP += 1
        else:
            G = 0
            B = 0
            for line in node.data:
                if line[-1] == "good":
                    G += 1
                else:
                    B += 1
            rateForGood = G / len(node.data)
            rateForBad = B / len(node.data)
            if row[-1] == "good":
                TP += rateForGood
                FP += rateForBad
            else:
                FN += rateForGood
                TN += rateForBad
        accuracy = (TP + TN) / len(table)
    Report = []
    Report.append("Accuracy:            " + str(accuracy))
    recall = TP/(TP+TN)
    Report.append("TPrate:              " + str(recall))
    Report.append("TNrate:              " + str(TN / (TP + TN)))
    presicion = TP / (TP + FP)
    Report.append("Presicion:           " + str(presicion))
    Report.append("F-Score:             " + str(2 * (presicion * recall) / (presicion + recall)))
    Report.append("Total number of TP:  " + str(TP))
    Report.append("Total number of TN:  " + str(TN))
    AccuracyRate = accuracy
    return Report


def randForest(train, test):
    # shuffle rows
    updatedTable = []
    random.shuffle(train)
    random.shuffle(test)
    # I merged test and train to suffle columns
    for line in train:
        updatedTable.append(list(line))
    for line in test:
        updatedTable.append(list(line))
    lenght = len(updatedTable[0])-1
    x = random.sample(range(lenght), lenght)
    x.append(lenght)
    updatedTable = [[row[i] for i in x] for row in updatedTable]
    # pick 4 attributes to find decision tree with them
    for i in range(5):
        max = len(updatedTable[0]) - 2
        rnd= random.randint(0, max)
        for line in updatedTable:
            line.pop(rnd)
    # split into two as train and test again
    train = list(updatedTable[:490])
    test = list(updatedTable[490:])
    # build a decision tree with train set
    root = DecisionTree(train, "All Records")
    buildNode(root)
    buildDecisionTree(root)
    global report
    # calculate accuracy with test set
    report = list(calculateAccuracy(root, test))
    return root
def main():
    # build tree for part A
    global table
    fileReader("trainSet.csv")
    root = DecisionTree(table, "All Records")
    buildNode(root)
    buildDecisionTree(root)
    # writing into a file as a report for part A
    file_path = 'reportPartA.txt'
    with open(file_path, 'w') as file:
        fileReader("trainSet.csv")
        Report = calculateAccuracy(root, table)
        file.write("Train Results \n")
        for line in Report:
            file.write(line+"\n")

        fileReader("testSet.csv")
        Report = calculateAccuracy(root, table)
        file.write("\nTest Results \n")
        for line in Report:
            file.write(line + "\n")

    # read datas into tables
    fileReader("trainSet.csv")
    trainTable = list(table)
    fileReader("testSet.csv")
    testTable = list(table)
    List = []
    global report
    # random forest 20 times it means that I checked for 20 decision tree
    for i in range(20):
        List.append(randForest(list(trainTable), list(testTable)))
        List.append(AccuracyRate)
        List.append(report)
    max = 0
    for i in range(0,60,3):
        if max < List[i+1]:
            max = List[i+1]
    # this root represents the optimal decision tree after random forest
    root = List[List.index(max)-1]
    # this is the report for optimal decision tree
    report = list(List[List.index(max)+1])
    file_path = 'reportPartB.txt'
    with open(file_path, 'w') as file:
        Report = calculateAccuracy(root, table)
        file.write("Test Results \n")
        for line in report:
            file.write(line + "\n")

if __name__ == '__main__':
    main()
