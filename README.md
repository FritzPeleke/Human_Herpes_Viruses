# Classification of Human Heerpes viruses into regions
This project has several scripts which should be ran with the stated versions. 
It has been noticed that scripts perfrom slightly differently on different operating systems.
## scripts
* Baseline model.py contains code for the k-nearest neighbour
* Network FM.py contains the code for deep network
* Genbank filefinder.py gets the genbank flatfiles from Genbank
* CDS extractor.py extracts coding sequences for major capsid protein
* data collector.py selects sequences > 600 nucleotides and cuts 400 nucleotide fragments.
* Bag of words.py creates the histograms
* plots.py plots the validation accuracy and loss from the .json files obtained from tensorboard.
* workthrough.py processes the coding sequences to create a histogram
* All data.zip contains all the datasets used for the project. It contains datasets before and after preprocessing.
## operating system
This project was performed on a windows 10 operating system. Scripts have however been seen to perform differently on mac operating system.
## Datasets
The first set of FASTA sequences found in Data folder was downloaded on 07.01.2020. The second set found in Data 2, were downloaded on 28.01.2020.
