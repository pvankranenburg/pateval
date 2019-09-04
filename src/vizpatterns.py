from music21 import *
import csv
import os
import shutil
import collections

#in shell do a: ulimit -Sn 10000

mtcann2path = '/Users/pvk/data/MTC/MTC-ANN-2.0.1/'
metadatapath = mtcann2path + 'metadata/'
#mtcpath = '/Users/pvk/Documents/Eigenwerk/Projects/MeertensTuneCollections/data/MTC-ANN-1.0/'
mtcfs1path = '//Users/pvk/data/MTC/MTC-FS-1.0/'

krnpath = '/Users/pvk/Documents/Eigenwerk/Projects/MGDP/TuneFamilies/corpus/allkrn'

# find out whether pattern should start at note before first index (c3 or c5)
def firstItemIsRelative(patternname):
	items = patternname.split("}{")
	if "c3" in items[0]: return True
	if "c5" in items[0]: return True
	if "diaintc" in items[0]: return True
	if "contour" in items[0]: return True
	return False

# color scheme from http://www.edwardtufte.com/bboard/q-and-a-fetch-msg?msg_id=0000HT
def getColor(element):
	if 'intref' in element: return 'blue'
	elif 'pitch' in element: return 'MediumSeaGreen'
	else: return 'red'
	
	#if 'intref' in element: return '0.0 0.45 0.7' #blue
	#elif 'pitch' in element: return '0.0 0.61 0.45' #bluish green
	#else: return '0.9 0.62 0.0' #orange
	#if 'intref' in element: return '0.0 0.45 0.7' #blue
	#elif 'pitch' in element: return '0.34 0.7 0.91' #sky blue
	#else: return '0.83 0.37 0.0' #vermillion
	
	#if 'intref' in element: return '0.0 0.0 1.0' #blue
	#elif 'pitch' in element: return '0.24 0.70 0.44' #MediumSeaGreen
	#else: return '1.0 0.0 0.0' #red

def formatFeaturelist2018(flist):
	res = ''
	d = flist.strip('{}').split(':')
	if not len(d[0]):
		return res
	for i in zip(d[0].split('*'),d[1].strip('[]').split(',')):
		res = res + i[0]+":"+i[1]+'; '
	if len(res)>2:
		res = res[1:-2]
	return res

def formatPatternName2018(name):
	its = name[1:-1].split('}{')
	res = '{'
	for it in its:
		f_it = formatFeaturelist2018(it)
		res = res + f_it + '}{'
	res = res[:-1]
	return res 

# reads pattern detection output from csv file
# [0] family G;
# [1] pattern X;
# [2] I(X);
# [3] -log(pval);
# [4] n(X,G)/n(G);
# [5] n(X,~G)/n(~G);
# [6] p(G|X);
# [7] instances;
# [8] ainstances
# returns list of dictionaries
# fields: name, occurrences, antioccurrences
def readPatterns(filename):
	patterns = []

	with open(filename,'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=';', quotechar='"')
		next(reader, None) # skip header
		for ix, row in enumerate(reader):
			name = row[1]
			firstIsRelative = False
			if firstItemIsRelative(name):
				name = "{}"+name
				firstIsRelative = True
			occ = {"corpus":row[0],
				   "name":formatPatternName2018(name),
				   "I(X)":row[2],
				   "-log(pval)":row[3],
				   "n(X,G)/n(G)":row[4],
				   "n(X,~G)/n(~G)":row[5],
				   "p(G|X)":row[6],
				   "occs":row[7],
				   "aoccs":row[8],
				   "firstIsRelative":firstIsRelative,
				   "index":str(ix).zfill(3)}
			patterns.append(occ)

	return patterns


def getAnnotatedNLBIDs():
    with open(metadatapath + 'MTC-ANN-tune-family-labels.csv') as f:
        result = [line.split(',')[0] for line in f.readlines()]
    return result

# take the name of a pattern and return the length
def getPatternLength(name):
	res = len(name.split("}{"))
	return res

# split the list of occurrences in single occurrences
# sort the occurrences according to NLB-id
# for format fma2016
def doGetOccurrences_fma2016(occs_asstring, pattern):
	adjust = 0
	if pattern["firstIsRelative"]:
		adjust = -1
	occs = []
	sp = occs_asstring.split("NLB")
	sp = sp[1:]
	sp = ["NLB"+x.strip() for x in sp]
	for occ in sp:
		spl = occ.split(' ')
		for item in spl[1:]:
			occs.append((spl[0],int(item)+adjust))
	return sorted(occs,key=lambda x:x[0])

# split the list of occurrences in single occurrences
# sort the occurrences according to NLB-id
# for file send nov 2017
def doGetOccurrences_nov2017(occs_asstring, pattern):
	adjust = 0
	if pattern["firstIsRelative"]:
		adjust = -1
	occs = []
	sp = occs_asstring.split("NLB")
	sp = sp[1:]
	sp = ["NLB"+x.strip() for x in sp]
	for occ in sp:
		nlb = occ.split(':')[0].strip()
		spl = occ.split(':(')[1].split(',')
		for item in spl:
			occs.append((nlb,int(item.strip(')'))+adjust))
	return sorted(occs,key=lambda x:x[0])

doGetOccurrences = doGetOccurrences_fma2016

# split the list of corpus occurrences in single occurrences
def getOccurrences(pattern):
	return doGetOccurrences(pattern["occs"], pattern)

# split the list of anti-corpus occurrences in single occurrences
def getAntiOccurrences(pattern):
	return doGetOccurrences(pattern["aoccs"], pattern)

# return the tune family name for a given NLB identfier
def getTuneFamily_mtcann2(NLBid):
	"""
	This function takes a NLBid of the pattern to be matched (string - 'NLBxxxxxx_yy'),
	and returns the name of the tune family

	Example: getTuneFamily('NLB073862_01')
	--> 'Er_woonde_een_vrouwtje_al_over_het_bos'
	"""
	filename = mtcann2path+"/metadata/MTC-ANN-tune-family-labels.csv"
	tf = "UNDEFINED"
	with open(filename) as f:
		doc = csv.reader(f, delimiter=",")
		for line in doc :
			if line[0] == NLBid :
				tf = line[1]
	return tf

def getTuneFamily_mtcfs1(NLBid):
	"""
	This function takes a NLBid of the pattern to be matched (string - 'NLBxxxxxx_yy'),
	and returns the name of the tune family

	Example: getTuneFamily('NLB073862_01')
	--> 'Er woonde een vrouwtje al over het bos', '9749_0'
	"""
	filename = mtcfs1path+"/metadata/MTC-FS.csv"
	tf = "UNDEFINED"
	tf_id = "UNDEFINED"
	with open(filename) as f:
		doc = csv.reader(f, delimiter=",", quotechar='"')
		for line in doc :
			if line[0] == NLBid:
				tf_id = line[13]
				tf = line[14]
	return tf, tf_id

# first look in MTC-ANN. If not found, look in MTC-FS
def getTuneFamily(NLBid):
	tf = getTuneFamily_mtcann2(NLBid)
	tf_id = ""
	if tf == "UNDEFINED":
		tf, tf_id = getTuneFamily_mtcfs1(NLBid)
	return tf, tf_id

# parse the score of a song
# add indices to the notes (tied notes counted as one)
def getScoreWithIndices(nlbid):
	# load song
	# add attribute to continuations of tied notes
	# remove lyrics
	# returns song
	s = converter.parse(krnpath+'/'+nlbid+'.krn')
	for n in s.flat.notes:
		n.lyric = None
		n.patterncount = 0
		if not n.tie:
			n.skip = False
		else:
			if n.tie.type == 'start':
				n.skip = False
			else:
				n.skip = True
	return s


# Returns the score with the pattern highlighted 
def annotatePatternInScore(occ, name, score, samecolor=False, addPatternCount=False, addNoteIndex=False):
	elements = name.split("}{")
	s = score.flat.notes
	notes = [n for n in s if not n.skip]
	for ix in occ[1:]:
		for note_ix, _ in enumerate(notes):
			if ix == note_ix:
				for el_ix, el in enumerate(elements):
					color = "red" #default color
					#color = '#009E73'
					if not samecolor:
						color = getColor(el)
					if note_ix+el_ix >= len(notes):
						print("Error: "+occ[0]+" "+name)
						continue
					n = notes[note_ix+el_ix]
					n.style.color = color
					n.patterncount += 1
					n.lyric = None
					if addPatternCount: n.addLyric(str(n.patterncount))	
				if addNoteIndex: notes[note_ix].addLyric(str(note_ix))

# Returns dictionary with all music21-scores containing the pattern
# and pattern occurrences indicated with colored notes
def annotatePattern(occs,name,scores=None,samecolor=False,addPatternCount=False,addNoteIndex=False):
	#build dictionary of music21 scores
	if scores == None:
		scores = collections.OrderedDict()
	for occ in occs:
		#create score if necessary
		if not occ[0] in scores:
			scores[occ[0]] = getScoreWithIndices(occ[0])
			scores[occ[0]].insert(metadata.Metadata())
			scores[occ[0]].metadata.title = occ[0]
		annotatePatternInScore(occ, name, scores[occ[0]], samecolor=samecolor, addPatternCount=addPatternCount,addNoteIndex=addNoteIndex)

	return scores

# make sure directory f exists
def ensure_dir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)

# create pngs of all scores
def writescores(scores, outputdir, max_number=0, prefix=""):
	if max_number == 0 : max_number = len(scores.values())
	for ix, s in enumerate(scores.values()):
		if ix < max_number:
			out = s.write('lily.png')
			shutil.move(out,outputdir+'/'+prefix+s.metadata.title+'.png')

# create a html file including all songs in which pattern occurs
# if pattern provided, print pattern info
def create_html(pattern_name, scores, ascores, outputdir, pattern=None, occ_prefix="", aocc_prefix=""):
	f = open(outputdir+'/index.html','w')
	f.write("<html>")
	f.write("<body>")
	f.write("<h3>"+pattern_name+"</h3>")
	
	if pattern:
		f.write("index : "+pattern['index']+"<br>")
		f.write("corpus : "+pattern["corpus"]+"<br>")
		f.write("I(X) : "+pattern["I(X)"]+"<br>")
		f.write("-log(pval) : "+pattern["-log(pval)"]+"<br>")
		f.write("n(X,G)/n(G) : "+pattern["n(X,G)/n(G)"]+"<br>")
		f.write("n(X,~G)/n(~G) : "+pattern["n(X,~G)/n(~G)"]+"<br>")
		f.write("p(G|X) : "+pattern["p(G|X)"]+"<br>")

	scores_lines = collections.defaultdict(lambda: collections.defaultdict(list))

	for s in scores.values():
		tf, tf_id = getTuneFamily(s.metadata.title)
		scores_lines[tf][s.metadata.title].append("<p>"+s.metadata.title+"</p>")
		scores_lines[tf][s.metadata.title].append("<p>"+tf+" "+tf_id+"</p>")
		scores_lines[tf][s.metadata.title].append("<p><img src=\""+occ_prefix+s.metadata.title+".png\"></p>")	

	for tf in sorted(scores_lines.keys()):
		for s in sorted(scores_lines[tf].keys()):
			for l in scores_lines[tf][s]:
				f.write(l)
				f.write('\n')

	f.write("<hr>")

	f.write("<p>Occurrences in "+str(len(ascores.values()))+" none tf members. Showing first max XXX.</p>")

	ascores_lines = collections.defaultdict(lambda: collections.defaultdict(list))

	for s in ascores.values():
		tf, tf_id = getTuneFamily(s.metadata.title)
		ascores_lines[tf][s.metadata.title].append("<p>"+s.metadata.title+"</p>")
		ascores_lines[tf][s.metadata.title].append("<p>"+tf+" "+tf_id+"</p>")
		ascores_lines[tf][s.metadata.title].append("<p><img src=\""+aocc_prefix+s.metadata.title+".png\"></p>")	

	for tf in sorted(ascores_lines.keys()):
		for s in sorted(ascores_lines[tf].keys()):
			for l in ascores_lines[tf][s]:
				f.write(l)
				f.write('\n')

	f.write("</body")
	f.write("</html>")
	f.close()

# create an index with a link to occurrences of each pattern
def create_html_index(patterns, outputdir):
	f = open(outputdir+'/index.html','w')
	f.write("<html>")
	f.write("<body>")
	f.write("<p><a href=\"all/index.html\">all</a></p>")
	for p in patterns:
		p_dir = './'+p['index']+'/'
		f.write("<p><a href=\""+p['index']+"/index.html\">"+p['index']+"</a> " + p["corpus"] + " " + p["p(G|X)"] + " " + p["n(X,G)/n(G)"] + " " + p['name'] + "</p>")
	f.write("</body")
	f.write("</html>")
	f.close()

# This is the main function to be called for each pattern
# will get the scores with pattern occurrences annotate
# put the images in a directory
# generate an html files with index
# separate patterns from corpus and anticorpus
# do not generate patterns from anticorpus if number is > max_aoccs_factor * number of occurrences in corpus
def viz_one_pattern(pattern, outputdir, max_number_aocc=0):
	do_aoccs = True
	occs = getOccurrences(pattern)
	aoccs = getAntiOccurrences(pattern)
	length = getPatternLength(pattern['name'])
	scores = annotatePattern(occs, pattern['name'],addNoteIndex=True)
	ascores = annotatePattern(aoccs, pattern['name'])
	ensure_dir(outputdir)
	writescores(scores,outputdir)
	writescores(ascores,outputdir,max_number=max_number_aocc)
	create_html(pattern['name'], scores, ascores, outputdir, pattern=pattern)


# This is the main function to be called for each pattern
# will get the scores with pattern occurrences annotate
# put the images in a directory
# generate an html file with index
# separate patterns from corpus and anticorpus
def viz_one_pattern_inonescore(pattern, scores, ascores):
	occs = getOccurrences(pattern)
	aoccs = getAntiOccurrences(pattern)
	length = getPatternLength(pattern['name'])
	scores = annotatePattern(occs, pattern["name"], scores,samecolor=True, addPatternCount=True)
	ascores = annotatePattern(aoccs, pattern["name"], ascores,samecolor=True, addPatternCount=True)


# This takes a csv, generates visualizations and html for each pattern
def vizpatterns(filename,output_base_dir, startat=0, max_number_aocc=15):
	patterns = readPatterns(filename)
	for p in patterns:
		if int(p['index']) < startat:
			continue
		outputdir = output_base_dir+'/'+p['index']+'/'
		viz_one_pattern(p,outputdir,max_number_aocc=max_number_aocc)
	create_html_index(patterns,output_base_dir+'/')

# This takes a csv, generates visualizations and html for each pattern. For each song, all pattern occurrences are summarized in one score. 
def vizpatterns_inonescore(filename,output_base_dir, startat=0):
	patterns = readPatterns(filename)
	#put all songs in both dictionaries, to also have the songs that have no patterns at all
	scores = {}
	for nlbid in getAnnotatedNLBIDs():
		scores[nlbid] = getScoreWithIndices(nlbid)
		scores[nlbid].insert(metadata.Metadata())
		scores[nlbid].metadata.title = nlbid
	ascores = {}
	for nlbid in getAnnotatedNLBIDs():
		ascores[nlbid] = getScoreWithIndices(nlbid)
		ascores[nlbid].insert(metadata.Metadata())
		ascores[nlbid].metadata.title = nlbid
	outputdir = output_base_dir+'/all/'
	for p in patterns:
		if int(p['index']) < startat:
			continue
		viz_one_pattern_inonescore(p,scores,ascores)
	ensure_dir(outputdir)
	writescores(scores,outputdir,prefix="c")
	writescores(ascores,outputdir,prefix="a")
	create_html("all",scores, ascores, outputdir, pattern=None, occ_prefix="c", aocc_prefix="a")

####################################
### MOTIFS

# ex = []
# with open('selected_examples.txt','r') as f:
# 	reader = csv.reader(f, delimiter=' ')
# 		for row in reader:
# 		ex.append(row)
# for e in ex:
# 	print e
# 	motif2ly('mtc_n5.csv',e[1],e[2],int(e[3]),outputdir='figs/',before=int(e[4]),after=int(e[5]))

# make lilypond source for a part of a score
# filename: pattern discovery output
# patternix: index of the pattern to be highlighted
# nlbid: nlbid of the song which contains pattern occurrence
# startix: index of first note (ties resolved) of the pattern occurrence
# outputdir: where to put the lily
# filenameprefix: prefix for lilypond filename
# before: number of notes before occurrence to include
# after: number of notes after occurrence to include
def motif2ly(filename, patternix, nlbid, startix, outputdir='.', filenameprefix='', before=2, after=2):
	patterns = readPatterns(filename)
	p = [pt for pt in patterns if pt['index'] == patternix][0]
	s = getScoreWithIndices(nlbid)
	pat_length = len(p['name'].split('}{'))
	annotatePatternInScore([nlbid, startix], p['name'], s, samecolor=False, addPatternCount=False, addNoteIndex=False)

	notes = [n for n in s.flat.notes if not n.skip]
	ix_first = max(0, startix-before)
	ix_last = min(len(notes)-1, startix+pat_length+after-1)
	off_first = notes[ix_first].getOffsetBySite(s.flat.notes)
	off_last = notes[ix_last].getOffsetBySite(s.flat.notes)

	#append tied notes
	n = s.flat.notes.getElementsByOffset(off_last)[0]
	if n.tie:
		off_last = s.flat.notes.getElementAfterElement(n, [note.Note]).getOffsetBySite(s.flat.notes)

	motif = s.flat.getElementsByOffset(off_first, off_last, includeEndBoundary=True)
	if ix_first > 0:
		motif.shiftElements(-motif[0].offset+motif.notes[0]._getMeasureOffset())
		try:
			motif.insert(0,motif[0].getContextByClass('TimeSignature'))
			motif.insert(0,motif[0].getContextByClass('KeySignature'))
		except exceptions21.StreamException:
			pass
		mm = motif.makeMeasures(finalBarline='final')
		mm[0].paddingLeft = motif.notes[0]._getMeasureOffset()
		motif = mm
	lyfile = motif.write('lily.ly')
	ensure_dir(outputdir)
	shutil.move(lyfile,outputdir+'/'+filenameprefix+nlbid+'_'+patternix+'_'+str(startix).zfill(3)+'.ly')

