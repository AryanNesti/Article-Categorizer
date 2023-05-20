import csv
from operator import not_
bruh_sports = []
bruh_business = []
bruh_politics = []
bruh_entertainment = []
bruh_tech=[]
not_sports = []
not_business = []
not_politics = []
not_entertainment = []
not_tech = []

# Splitting the cateegories
with open("news-train.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        if row[2] == 'sport':
          bruh_sports.append(row)
        else:
            not_sports.append(row)
        if row[2] == 'business':
          bruh_business.append(row)
        else:
           not_business.append(row) 
        if row[2] == 'politics':
          bruh_politics.append(row)
        else:
            not_politics.append(row)
        if row[2] == 'entertainment':
          bruh_entertainment.append(row)
        else:
            not_politics.append(row)
        if row[2] == 'tech':
          bruh_tech.append(row)
        else:
            not_tech.append(row)




# Writing files for each category
with open('business.txt','w') as file:
  for x in bruh_business:
    file.write(str(x[1])+"\n")

with open('sport.txt','w') as file:
  for x in bruh_sports:
    file.write(str(x[1]) +"\n")

with open('politics.txt','w') as file:
  for x in bruh_politics:
    file.write(str(x[1])+"\n")

with open('tech.txt','w') as file:
  for x in bruh_tech:
    file.write(str(x[1])+"\n")

with open('entertainment.txt','w') as file:
  for x in bruh_entertainment:
    file.write(str(x[1])+"\n")
    
# Writing text files contianing 4 categories excluding the listed category
with open('not_sports.txt','w') as file:
  for x in not_sports:
    file.write(str(x[1])+"\n")

with open('not_tech.txt','w') as file:
  for x in not_tech:
    file.write(str(x[1])+"\n")

with open('not_entertainment.txt','w') as file:
  for x in not_entertainment:
    file.write(str(x[1])+"\n")

with open('not_politics.txt','w') as file:
  for x in not_politics:
    file.write(str(x[1])+"\n")

with open('not_business.txt','w') as file:
  for x in not_business:
    file.write(str(x[1])+"\n")