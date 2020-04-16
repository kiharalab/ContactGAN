import heapq
import json
import os
import numpy as np

def create_dir(directory):

	if not os.path.exists(directory):
		os.makedirs(directory)

def read_target_list(list_fn):

	targets = []

	f = open(list_fn, 'r')
	for line in f:
	    targets.append(line.strip())

	f.close()
	
	return targets

def get_separation(mode):

	if mode == 'long':
		return 24

	if mode == 'medium+long':
		return 12


def get_real_contacts(contact_map,mode):

	contacts = {}

	separation = get_separation(mode)

	for i in range(len(contact_map)):
		for j in range(len(contact_map[0])):
			if abs(i - j) >= separation:
				if contact_map[i][j] == 1:
					if i+1 in contacts:
						contacts[i+1].append(j+1)
					else:
						contacts[i+1] = [j+1]

	return contacts

def contact_accuracy(real_contacts,predicted_contacts):

	correct = 0
	incorrect = 0

	#Compare the two dict
	for i in predicted_contacts:
		if i in real_contacts:
			c = predicted_contacts[i]
			for j in c:
				if j in real_contacts[i]:
					correct += 1
				else:
					incorrect += 1

		else:
			incorrect += len(predicted_contacts[i])

	accuracy = float(correct)/float(correct + incorrect)
	
	return accuracy,correct,incorrect

def contact_accuracy_ss(real_contacts,predicted_contacts,ss_list):

	correct = 0
	incorrect = 0

	#Compare the two dict
	aacnt=0
	bbcnt=0
	abcnt=0
	for i in predicted_contacts:
		if i in real_contacts:
			c = predicted_contacts[i]
			for j in c:
				if j in real_contacts[i]:
					correct += 1
				else:
					incorrect += 1

				idx_x, idx_y = i-1, j-1
				if j in real_contacts[i] and ss_list[idx_x]==2 and ss_list[idx_y]==2:
					aacnt+=1

				elif j in real_contacts[i] and ss_list[idx_x]==1 and ss_list[idx_y]==1:
					bbcnt+=1

				elif j in real_contacts[i] and ((ss_list[idx_x]==1 and ss_list[idx_y]==2) or (ss_list[idx_x]==2 and ss_list[idx_y]==1)):
				    abcnt+=1

		else:
			incorrect += len(predicted_contacts[i])

	accuracy = float(correct)/float(correct + incorrect)
	
	return accuracy, aacnt, bbcnt, abcnt


def load_test_contacts(mode):

	if mode == 'long':
		file = 'casp13_L_contacts.json'

	if mode == 'medium_and_long':
		file = 'casp13_ML_contacts.json'

	contacts = {}

	path = '/net/kihara/home/jain163/Desktop/Projects/folding/Z_Project/casp13/casp13_contact_map/'

	with open(path+file) as f:
		data = json.load(f)

	return data

def load_domains():

	domains = {}

	path = '/net/kihara/home/jain163/Desktop/Projects/folding/Z_Project/casp13/casp13_contact_map/'
	file = 'domains_evaluated_corrected'
	f = open(path+file,'r')

	for row in f:

		r = row.strip()
		t = r.split()

		if t[1] == 'evaluated':

			tmp = t[0].split(':')
			target = tmp[0].split('-')[0]
			domain_num = tmp[0].split('-')[1]
			seq_num = tmp[1]
			if target not in domains:
				domains[target] = {}
			domains[target][domain_num] = seq_num

	return domains

def inside_domain(domain_aa,i):
	#Check if i is within a domain
	domain_seqeunces = domain_aa.split(',')

	for cur_range in domain_seqeunces:

		low = int(cur_range.split('-')[0])
		high = int(cur_range.split('-')[1])

		if int(i) >= low and int(i) <= high:
			return True #If i is within any range, it is accepted

	return False


def get_predicted_contacts(contact_map,mode,l_by, domain_aa):

	contacts = {}

	separation = get_separation(mode)

	domain_seqeunces = domain_aa.split(',')
	total_length = 0
	for cur_range in domain_seqeunces:
		low = int(cur_range.split('-')[0])
		high = int(cur_range.split('-')[1])
		length = high - low + 1
		total_length = total_length + length

	l = int(total_length / l_by)

	contact_heap = []
	for i in range(len(contact_map)):
		for j in range(len(contact_map[0])):
			if abs(i - j) >= int(separation):
				val = -contact_map[i][j] #Doing negative to create a max heap
				heapq.heappush(contact_heap,(val,i+1,j+1))

	while ( l > 0):

		if len(contact_heap) > 0:
			prob,r,c = heapq.heappop(contact_heap)
			if inside_domain(domain_aa,r) and inside_domain(domain_aa,c): #Check both amino acid are inside domain
				if c not in contacts or r not in contacts[c]: # For i,j check if j,i is not already included

					if r in contacts:
						contacts[r].append(c)
					else:
						contacts[r] = [c]
					l = l - 1
		else:
			contacts[99999] = []
			while(l>0):
				contacts[99999].append(l)
				l = l - 1

	return contacts