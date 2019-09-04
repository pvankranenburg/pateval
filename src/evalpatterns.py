#####################################################################################
#####################################################################################

# Terminology
# Motif: annotated motif (class ; occurrences)
# Pattern: discovered pattern (name / ix ; occurrences)

# Collins
# PPP: Ground Truth Patterns, PI
# PP: Ground Truth Pattern, curly P
# P: Ground Truth Pattern occurrence, P
# mP: Number of Ground Truth Pattern occurrences of pattern P
# nPP: Number of Ground Truth Patterns
# QQQ: Discovered Patterns, KSI
# QQ: Discovered Pattern, curly Q
# Q: Discovered Pattern occurrence Q 
# mQ: number of Discoverend Pattern occurrences of pattern Q
# mQQ: number of Discovered Patterns

#####################################################################################
#####################################################################################

import collections
import csv
import numpy as np
from music21 import *


#####################################################################################
#####################################################################################
# Config

# global variable... brrrrr....
mtcann2path = '/Users/pvk/data/MTC/MTC-ANN-2.0.1/'

tuneFamAbbr = {"Heer": "Daar_ging_een_heer_1", "Jonkheer": "Daar_reed_een_jonkheer_1", "Ruiter2": "Daar_was_laatstmaal_een_ruiter_2", "Maagdje": "Daar_zou_er_een_maagdje_vroeg_opstaan_2", "Dochtertje": "Een_Soudaan_had_een_dochtertje_1", "Lindeboom": "Een_lindeboom_stond_in_het_dal_1", "Zoeteliefjes": "En_er_waren_eens_twee_zoeteliefjes", "Ruiter1": "Er_reed_er_eens_een_ruiter_1", "Herderinnetje": "Er_was_een_herderinnetje_1", "Koopman": "Er_was_een_koopman_rijk_en_machtig", "Meisje": "Er_was_een_meisje_van_zestien_jaren_1", "Vrouwtje": "Er_woonde_een_vrouwtje_al_over_het_bos", "Femmes": "Femmes_voulez_vous_eprouver", "Halewijn2": "Heer_Halewijn_2", "Halewijn4": "Heer_Halewijn_4", "Stavoren": "Het_vrouwtje_van_Stavoren_1", "Zomerdag": "Het_was_laatst_op_een_zomerdag", "Driekoningenavond": "Het_was_op_een_driekoningenavond_1", "Stad": "Ik_kwam_laatst_eens_in_de_stad", "Stil": "Kom_laat_ons_nu_zo_stil_niet_zijn_1", "Schipper": "Lieve_schipper_vaar_me_over_1", "Nood": "O_God_ik_leef_in_nood", "Soldaat": "Soldaat_kwam_uit_de_oorlog", "Bruidje": "Vaarwel_bruidje_schoon", "Verre": "Wat_zag_ik_daar_van_verre_1", "Boom": "Zolang_de_boom_zal_bloeien_1"}
melodyLengths = {'NLB070238_01': 78, 'NLB070463_01': 48, 'NLB072450_01': 47, 'NLB073286_01': 50, 'NLB075313_01': 56, 'NLB072256_01': 31, 'NLB072085_01': 67, 'NLB070801_01': 55, 'NLB074003_01': 31, 'NLB072254_01': 38, 'NLB073754_02': 34, 'NLB072585_01': 48, 'NLB074333_01': 30, 'NLB073287_02': 72, 'NLB073287_01': 74, 'NLB074334_01': 33, 'NLB138219_01': 47, 'NLB072356_02': 38, 'NLB074754_01': 33, 'NLB072627_01': 75, 'NLB072355_12': 36, 'NLB072378_01': 34, 'NLB073946_01': 41, 'NLB072255_01': 33, 'NLB075064_01': 49, 'NLB147463_01': 25, 'NLB075525_01': 81, 'NLB070411_01': 33, 'NLB144100_01': 59, 'NLB073269_02': 50, 'NLB075063_01': 62, 'NLB072912_01': 50, 'NLB072253_01': 35, 'NLB073997_01': 34, 'NLB074948_01': 57, 'NLB072306_01': 66, 'NLB073628_01': 67, 'NLB071944_01': 48, 'NLB075018_02': 46, 'NLB146741_01': 83, 'NLB143240_01': 32, 'NLB072154_01': 32, 'NLB072559_01': 62, 'NLB072299_01': 62, 'NLB075074_01': 40, 'NLB073991_02': 46, 'NLB071478_01': 38, 'NLB076495_01': 92, 'NLB072250_01': 31, 'NLB072357_01': 36, 'NLB162519_01': 58, 'NLB075848_01': 72, 'NLB070141_01': 33, 'NLB075065_01': 39, 'NLB074038_01': 85, 'NLB072500_01': 69, 'NLB074342_01': 38, 'NLB072714_01': 35, 'NLB075021_01': 65, 'NLB076118_01': 49, 'NLB072237_01': 55, 'NLB070748_01': 50, 'NLB074390_01': 49, 'NLB146608_01': 32, 'NLB073486_01': 40, 'NLB073992_01': 64, 'NLB074234_01': 46, 'NLB076426_01': 34, 'NLB073404_01': 55, 'NLB070144_01': 31, 'NLB072288_01': 46, 'NLB073990_01': 46, 'NLB074227_01': 44, 'NLB075318_01': 83, 'NLB072289_01': 56, 'NLB073804_02': 35, 'NLB073822_01': 41, 'NLB071957_03': 30, 'NLB075035_01': 65, 'NLB075309_02': 83, 'NLB071369_01': 47, 'NLB076128_01': 51, 'NLB073210_01': 57, 'NLB112210_01': 40, 'NLB134389_01': 47, 'NLB111656_01': 40, 'NLB073639_01': 43, 'NLB074349_01': 35, 'NLB073994_01': 34, 'NLB070096_01': 44, 'NLB075325_02': 52, 'NLB075325_01': 38, 'NLB073803_01': 35, 'NLB070521_01': 77, 'NLB072565_01': 32, 'NLB076303_01': 37, 'NLB072283_01': 44, 'NLB075034_01': 49, 'NLB073888_01': 46, 'NLB070122_01': 46, 'NLB070532_01': 30, 'NLB073588_01': 41, 'NLB074603_01': 22, 'NLB073296_01': 38, 'NLB072664_01': 44, 'NLB070526_01': 49, 'NLB071958_01': 34, 'NLB147912_01': 33, 'NLB141314_01': 43, 'NLB075367_01': 40, 'NLB075158_01': 70, 'NLB074286_01': 88, 'NLB074840_02': 37, 'NLB074840_01': 36, 'NLB072355_02': 32, 'NLB072355_01': 23, 'NLB162684_01': 62, 'NLB075057_01': 83, 'NLB075156_01': 70, 'NLB112115_01': 82, 'NLB073897_01': 30, 'NLB075431_01': 57, 'NLB074100_01': 29, 'NLB074649_01': 42, 'NLB072837_01': 67, 'NLB074166_01': 37, 'NLB075013_01': 54, 'NLB152784_01': 70, 'NLB075040_01': 42, 'NLB075616_01': 38, 'NLB071014_01': 49, 'NLB146728_01': 81, 'NLB072823_01': 41, 'NLB074246_02': 68, 'NLB073225_02': 32, 'NLB073225_01': 30, 'NLB076076_01': 52, 'NLB145856_01': 30, 'NLB074246_01': 33, 'NLB074533_01': 41, 'NLB072553_01': 32, 'NLB073777_01': 41, 'NLB076211_01': 43, 'NLB134480_01': 32, 'NLB072895_01': 64, 'NLB072851_02': 45, 'NLB072851_01': 44, 'NLB070033_01': 42, 'NLB074672_01': 36, 'NLB073333_01': 31, 'NLB074938_01': 37, 'NLB072665_01': 40, 'NLB074276_01': 36, 'NLB072897_01': 30, 'NLB074769_01': 37, 'NLB075073_01': 36, 'NLB075059_01': 41, 'NLB075191_01': 33, 'NLB073331_01': 44, 'NLB075906_01': 48, 'NLB072920_01': 35, 'NLB072359_01': 23, 'NLB074583_01': 46, 'NLB070360_01': 55, 'NLB070535_01': 69, 'NLB146699_01': 56, 'NLB074277_01': 42, 'NLB073681_01': 46, 'NLB075635_01': 35, 'NLB070079_01': 57, 'NLB074048_02': 39, 'NLB073709_01': 34, 'NLB073339_01': 81, 'NLB070078_01': 69, 'NLB072285_01': 35, 'NLB111484_01': 50, 'NLB073337_02': 44, 'NLB072358_01': 37, 'NLB072103_01': 57, 'NLB074427_01': 41, 'NLB074260_02': 39, 'NLB074260_01': 26, 'NLB074552_01': 46, 'NLB072883_01': 67, 'NLB075176_01': 32, 'NLB073304_01': 57, 'NLB111478_01': 83, 'NLB124573_01': 51, 'NLB072284_01': 46, 'NLB070693_01': 59, 'NLB074261_01': 40, 'NLB075551_01': 45, 'NLB075739_03': 53, 'NLB075379_01': 83, 'NLB074156_03': 32, 'NLB073626_01': 52, 'NLB072286_01': 50, 'NLB075174_01': 34, 'NLB076740_01': 58, 'NLB074547_02': 60, 'NLB073879_02': 37, 'NLB074104_01': 37, 'NLB151180_01': 57, 'NLB073326_01': 37, 'NLB071669_01': 40, 'NLB073958_01': 38, 'NLB071082_01': 48, 'NLB072287_01': 42, 'NLB074452_01': 35, 'NLB074956_01': 41, 'NLB073895_01': 38, 'NLB162526_01': 68, 'NLB073788_01': 53, 'NLB070089_01': 33, 'NLB072360_01': 36, 'NLB072015_01': 43, 'NLB152778_01': 56, 'NLB073998_01': 59, 'NLB076271_01': 98, 'NLB070606_01': 55, 'NLB075249_01': 32, 'NLB075612_01': 46, 'NLB073775_02': 38, 'NLB075184_01': 58, 'NLB134474_01': 43, 'NLB074443_01': 58, 'NLB145525_01': 64, 'NLB112123_01': 55, 'NLB070732_01': 52, 'NLB074962_01': 33, 'NLB075831_01': 37, 'NLB074575_01': 66, 'NLB072967_01': 59, 'NLB072754_01': 37, 'NLB073324_01': 51, 'NLB073743_01': 39, 'NLB074613_02': 35, 'NLB072946_01': 35, 'NLB074216_01': 35, 'NLB072248_01': 36, 'NLB072708_01': 45, 'NLB073750_01': 33, 'NLB074161_01': 62, 'NLB076632_01': 38, 'NLB073393_01': 66, 'NLB072638_01': 44, 'NLB072457_01': 40, 'NLB141648_01': 32, 'NLB072721_01': 37, 'NLB073311_01': 62, 'NLB167193_01': 51, 'NLB074157_01': 64, 'NLB144072_01': 39, 'NLB073150_01': 45, 'NLB072003_01': 46, 'NLB075742_01': 67, 'NLB070839_01': 33, 'NLB070125_01': 53, 'NLB075167_01': 82, 'NLB075079_01': 27, 'NLB070134_01': 40, 'NLB072881_01': 37, 'NLB072567_01': 39, 'NLB146731_01': 84, 'NLB070740_01': 32, 'NLB074004_01': 41, 'NLB074954_01': 69, 'NLB072382_02': 38, 'NLB071666_01': 39, 'NLB072813_01': 71, 'NLB074308_01': 51, 'NLB074769_02': 40, 'NLB073939_01': 73, 'NLB074470_01': 45, 'NLB071016_01': 83, 'NLB073277_01': 47, 'NLB141407_01': 81, 'NLB073685_01': 36, 'NLB076258_01': 56, 'NLB074593_01': 52, 'NLB071227_01': 67, 'NLB072774_02': 44, 'NLB074378_02': 35, 'NLB071064_01': 31, 'NLB073076_01': 32, 'NLB111760_01': 46, 'NLB073929_01': 27, 'NLB074468_01': 50, 'NLB072441_01': 55, 'NLB073120_01': 36, 'NLB074309_01': 75, 'NLB071441_01': 49, 'NLB070137_01': 37, 'NLB072886_02': 62, 'NLB072886_01': 63, 'NLB144042_01': 39, 'NLB072871_01': 49, 'NLB072311_01': 63, 'NLB075307_03': 74, 'NLB073672_01': 44, 'NLB072898_01': 47, 'NLB072898_02': 45, 'NLB073866_01': 34, 'NLB125421_01': 85, 'NLB070996_01': 64, 'NLB076625_01': 42, 'NLB074077_02': 38, 'NLB075532_01': 38, 'NLB072862_01': 62, 'NLB072497_01': 77, 'NLB072587_02': 45, 'NLB074182_01': 30, 'NLB072587_01': 44, 'NLB073374_01': 50, 'NLB072647_01': 62, 'NLB074860_01': 33, 'NLB072499_01': 45, 'NLB072624_01': 69, 'NLB072482_01': 39, 'NLB125427_01': 47, 'NLB074028_01': 34, 'NLB015569_01': 48, 'NLB076130_07': 67, 'NLB070493_01': 48, 'NLB072968_01': 84, 'NLB075881_02': 59, 'NLB071237_01': 30, 'NLB072688_01': 34, 'NLB072257_01': 35, 'NLB073426_01': 37, 'NLB073516_01': 41, 'NLB074437_01': 44, 'NLB072614_01': 42, 'NLB075771_02': 73, 'NLB073862_01': 75, 'NLB072691_01': 33, 'NLB070475_01': 57, 'NLB073298_01': 45, 'NLB135273_01': 32, 'NLB072690_01': 40, 'NLB070492_01': 34, 'NLB075273_01': 38, 'NLB073046_01': 43, 'NLB073562_01': 76, 'NLB073483_01': 45, 'NLB074426_02': 34, 'NLB072505_01': 43, 'NLB074328_01': 45, 'NLB112233_01': 48, 'NLB141251_01': 50, 'NLB074336_01': 23, 'NLB075068_01': 35, 'NLB141649_01': 83, 'NLB073031_01': 58, 'NLB073146_01': 48, 'NLB073771_02': 57, 'NLB071974_02': 60, 'NLB074007_01': 83, 'NLB074433_01': 36, 'NLB070053_01': 50}
tuneFamliesPerAnnotator = {"Mariet": ["Zolang_de_boom_zal_bloeien_1", "Soldaat_kwam_uit_de_oorlog", "En_er_waren_eens_twee_zoeteliefjes", "En_er_waren_eens_twee_zoeteliefjes", "Het_was_op_een_driekoningenavond_1", "Heer_Halewijn_2", "Een_lindeboom_stond_in_het_dal_1"], "Marieke": ["Wat_zag_ik_daar_van_verre_1", "Er_was_een_koopman_rijk_en_machtig", "Ik_kwam_laatst_eens_in_de_stad", "Daar_ging_een_heer_1", "Een_Soudaan_had_een_dochtertje_1", "Er_woonde_een_vrouwtje_al_over_het_bos", "Heer_Halewijn_4", "Kom_laat_ons_nu_zo_stil_niet_zijn_1", "Het_vrouwtje_van_Stavoren_1", "O_God_ik_leef_in_nood"], "Ellen": ["Er_reed_er_eens_een_ruiter_1", "Lieve_schipper_vaar_me_over_1", "Het_was_laatst_op_een_zomerdag", "Daar_was_laatstmaal_een_ruiter_2", "Er_was_een_meisje_van_zestien_jaren_1", "Daar_reed_een_jonkheer_1", "Vaarwel_bruidje_schoon", "Er_was_een_herderinnetje_1", "Femmes_voulez_vous_eprouver", "Daar_zou_er_een_maagdje_vroeg_opstaan_2"]}

epsilon=0.0001

#####################################################################################
#####################################################################################
# Extension of defaultdict

# needed to pass key as argument to the default factory (i.e. length of the melody)
class keydefaultdict(collections.defaultdict):
	def __missing__(self, key):
		if self.default_factory is None:
			raise KeyError( key )
		else:
			ret = self[key] = self.default_factory(key)
			return ret


#####################################################################################
#####################################################################################
# Reading data

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

def getFullNameTuneFamily(abbr):
	return tuneFamAbbr[abbr]

def buildMelodyLengthTable():
	filename = mtcann2path+"/metadata/MTC-ANN-tune-family-labels.csv"
	lengths = {}
	with open(filename) as f:
		doc = csv.reader(f, delimiter=",")
		for line in doc:
			lengths[line[0]] = getMelodyLength(line[0])
	return lengths


def getMotifTable(fp=mtcann2path+'metadata/MTC-ANN-motifs.csv'):
	with open(fp, 'r') as f:
		reader = csv.reader(f)
		motifs = [row for row in reader]
	return motifs

# find out whether pattern should start at note before first index (contour / diaint)
def firstItemIsRelative(patternname):
	items = patternname.split("}{")
	if "c3" in items[0]: return True
	if "c5" in items[0]: return True
	if "diaintc" in items[0]: return True
	if "contour" in items[0]: return True
	return False

# take the name of a pattern and return the length
def getPatternLength(name):
	res = len(name.split("}{"))
	return res

# split the string with pattern occurrences into a list
# each element is tuple (nlbid, ix) E.g. ('NLB070996_01', 15)
def doGetOccurrences(occs_asstring, firstIsRelative, filterAnnotated=False):
	adjust = 0
	if firstIsRelative:
		adjust = -1
	occs = []
	sp = occs_asstring.split("NLB")
	sp = sp[1:]
	sp = ["NLB"+x.strip() for x in sp]
	for occ in sp:
		spl = occ.split(' ')
		if filterAnnotated:
			if not ( spl[0] in melodyLengths.keys() ):
				continue
		for item in spl[1:]:
			occs.append((spl[0],int(item)+adjust))
	return occs # N.B. make sure order is the same as input. First occurrences, then a-occurrences!!

# load a pattern table from file
# and return list of patterns
# each element is a dict describing a discovered pattern and occurrences
def getPatternTable(patternfile, filterAnnotated=False):
	patterns = []
	with open(patternfile,'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=';', quotechar='"')
		next(reader, None) # skip header
		for row in reader:
			name = row[1]
			# If the first item is relative to previous note, include previous note in pattern occurrences
			# By adding "{}" to the motif name
			firstIsRelative = False
			if firstItemIsRelative(name):
				name = "{}"+name
				firstIsRelative = True
			occs = row[7]
			aoccs = row[8]
			pat = {"name":name,
			       "occs":doGetOccurrences(occs, firstIsRelative, filterAnnotated),
			       "aoccs":doGetOccurrences(aoccs, firstIsRelative, filterAnnotated),
			       "tunefamily": getFullNameTuneFamily(row[0]),
			       "length":getPatternLength(name)}
			patterns.append(pat)

	return patterns


#####################################################################################
#####################################################################################
# Create Dictionaries

# return the number of notes in an ANN-2.0 melody (ties resovled)
def getMelodyLength(nlbid):
	#s = converter.parse(mtcann2path+'krn/'+nlbid+'.krn')
	#return len(s.flat.notes.stripTies())
	
	#faster:
	return melodyLengths[nlbid]

# return extract the motif index from a annotated motif class id. e.g. "3:bag" -> 3
def getMotifIx(motifid):
	return int(motifid.split(':')[0])

# construct a dict from the motiftable
# [nlbid] -> ndarray 
# the array has length of the melody. Elements correspond with notes.
# elements are zero if no (annotated) motif occur, and motif index if motif occur.
def motifTableToDict_pernlbid(motiftable):
	motifdict = keydefaultdict(lambda key: np.zeros(getMelodyLength(key), dtype=int))
	for row in motiftable:
		motif_ix = getMotifIx(row[9])
		motifdict[row[1]][int(row[6]):int(row[7])] = motif_ix
	return motifdict

# construct a dict from the patterntable
# [pattern][nlbid] -> ndarray
# key is the pattern index
# the array has length of the melody. elements correspond with notes.
# element is 1 if the pattern occurs at the note, zero otherwise
# if use_aoccs is False, only instances are incorporated in the dict
def patternTableToDict_perpattern(patterntable, use_aoccs=True):
	patterndict = collections.defaultdict(lambda: keydefaultdict(lambda key: np.zeros(getMelodyLength(key), dtype=int)))
	for ix, pattern in enumerate(patterntable):
		if use_aoccs:
			print "Using anti-instances"
			occs = pattern["occs"]+pattern["aoccs"]
		else:
			print "Not using anti-instances"
			occs = pattern["occs"]
		for occ in occs:
			patterndict[ix][occ[0]] [occ[1]:occ[1]+pattern["length"]] = 1
	return patterndict


#####################################################################################
#####################################################################################
# Perform evaluation measures

#use pattern occurence vector as mask for motif vector
#throw away zero elements
def getCoOccurrencesMelodies(patterns, motifs):
	pm = np.array(patterns)
	mm = np.array(motifs)
	mask = pm>0
	#return list( mm[mask][mm[mask]>0] )
	return list( mm[mask] )

# get a table of coocurrences of discovered patterns and annotated patterns
def getCoOccurrences(pdict, mdict):
	patcoocs = {}
	for pat in list(pdict.keys()):
		coocs = []
		for nlbid in list(pdict[pat].keys()):
			coocs = coocs + getCoOccurrencesMelodies(pdict[pat][nlbid], mdict[nlbid])
		patcoocs[pat] = coocs
	return patcoocs


#####################################################################################
#####################################################################################
# Establishment precision / recall

# Motifclass name is tune family name + motif class id
def motifClassName(tunefamily,motifclassname):
	return tunefamily+'-'+str(getMotifIx(motifclassname)).zfill(2)

# for these we need another motif dict with motif class as key
# [class] -> occurrences
# occurrences is tuple (nlbid, startix, length)
def motifTableToDict_perclass_occs(motiftable):
	PPP = collections.defaultdict(list)
	for row in motiftable:
		mclass = motifClassName(row[0], row[9])
		PPP[mclass].append( (row[1], int(row[6]), int(row[8]) ) )
	return PPP

# [pattern] -> occurrences
# occurrences is tuple (nlbid, startix, length)
def patternTableToDict_perpattern_occs(patterntable, use_aoccs=True):
	QQQ = collections.defaultdict(list)
	for ix, pattern in enumerate(patterntable):
		if use_aoccs:
			occs = pattern["occs"]+pattern["aoccs"]
		else:
			occs = pattern["occs"]
		for occ in occs:
			QQQ[ix].append( (occ[0], int(occ[1]), int( pattern["length"] ) ) )
	return QQQ

# Filter for Tune Family
# FOR NOW: only look at first occurrence. All occurrences are supposed to be in same tunefamily
def PPP_forTuneFamilies(PPP, tfs):
	keys = []
	for p in list(PPP.keys()):
		if getTuneFamily_mtcann2(PPP[p][0][0]) in tfs:
			keys.append(p)
	return dict( [(i, PPP[i]) for i in keys if i in PPP] )

def QQQ_forTuneFamilies(QQQ, tfs, ptable):
	keys = []
	for q in list(QQQ.keys()):
		if ptable[q]['tunefamily'] in tfs:
			keys.append(q)
	return dict( [(i, QQQ[i]) for i in keys if i in QQQ] ) 

# Cardinalityscore sc
# both Pi and Qi are tuples
# (nlbix, startix, length)
def sc(Pi,Qi):
	Qi_set = set(range(Qi[1],Qi[1]+Qi[2]))
	Pi_set = set(range(Pi[1],Pi[1]+Pi[2]))
	return float( len ( Qi_set & Pi_set ) ) / max( float(len(Qi_set)), float(len(Pi_set)) )

# Alternative similarity score
# Fraction of notes from groundtruth pattern Pi that are covered by discovered pattern Qi
# with max 1.0
def sc2(Pi,Qi):
	Qi_set = set(range(Qi[1],Qi[1]+Qi[2]))
	Pi_set = set(range(Pi[1],Pi[1]+Pi[2]))
	intersect = Qi_set & Pi_set
	score = float(len(intersect)) / float(len(Pi_set)) 
	return min(score, 1.0)

# 
# ?
#
def sc3(Pi,Qi):
	Qi_set = set(range(Qi[1],Qi[1]+Qi[2]))
	Pi_set = set(range(Pi[1],Pi[1]+Pi[2]))
	intersect = Qi_set & Pi_set
	score = 0.0
	if float(len(intersect)) > 0.5*float(len(Pi_set))-epsilon and len(Qi_set) <= len(Pi_set)+2:
		score = 1.0
	return score

#
# return 1 if all notes of discovered pattern Qi are part of groundtruth pattern Pi
# 
def sc4(Pi,Qi):
	Qi_set = set(range(Qi[1],Qi[1]+Qi[2]))
	Pi_set = set(range(Pi[1],Pi[1]+Pi[2]))
	if Qi_set.issubset(Pi_set):
		return 1.0
	else:
		return 0.0

# construct score matrix for discovered pattern PP and annotated motif QQ
# P is list of occurrences (tuple: (nlbix, startix, length) )
# Q is list of occurrences (tuple: (nlbix, startix, length) )
def build_s(PP,QQ,simfunc):
	mP = len(PP)
	mQ = len(QQ)
	res = np.zeros( (mP, mQ) )
	for i in range(mP):
		for j in range(mQ):
			if PP[i][0] == QQ[j][0]: # same melody
				res[i,j] = simfunc( PP[i], QQ[j] )
	return res

# Max in Score matrix (S(PP,QQ) in formaliation Collins)
def s_max(s):
	return np.max(s)

# construct establishment matrix
def build_S(PPP,QQQ,simfunc):
	nPP = len(PPP)
	nQQ = len(QQQ)
	#iterate over discovered patterns and motif classes
	#create matrix S
	S = np.zeros((nPP,nQQ))
	#fill
	for i in range(nPP):
		for j in range(nQQ):
			S[i,j] = s_max( build_s( PPP[list(PPP.keys())[i]], QQQ[list(QQQ.keys())[j]], simfunc ) )
	return S

# Precision of s
def precision_of_s(s):
	mQ = s.shape[1]
	return np.sum(np.max(s,0)) / float(mQ)

# Recall of s
def recall_of_s(s):
	mP = s.shape[0]
	return np.sum(np.max(s,1)) / float(mP)

# construct occurrence precision and occurrence recall matrices
def build_Op_and_Or(PPP,QQQ,S,simfunc,threshold=0.75):
	#find indeces of elements of S > threshold
	x,y  = np.where(S>threshold)
	I = list(zip(x,y))
	nPP = len(PPP)
	nQQ = len(QQQ)
	Op = np.zeros((nPP,nQQ))
	Or = np.zeros((nPP,nQQ))
	for i,j in I:
		Op[i,j] = precision_of_s( build_s( PPP[list(PPP.keys())[i]], QQQ[list(QQQ.keys())[j]], simfunc ) )
		Or[i,j] = recall_of_s( build_s( PPP[list(PPP.keys())[i]], QQQ[list(QQQ.keys())[j]], simfunc ) )
	return Op, Or

def estPrecision(S):
	nQQ = S.shape[1]
	return np.sum(np.max(S,0)) / float(nQQ)

def estRecall(S):
	nPP = S.shape[0]
	return np.sum(np.max(S,1)) / float(nPP) 

def estF1(estP,estR):
	return 2.0*estP*estR / (estP + estR)

def occPrecision(Op):
	return np.mean(np.extract(np.max(Op,0)>0, np.max(Op,0)))

def occRecall(Or):
	return np.mean(np.extract(np.max(Or,1)>0, np.max(Or,1)))

def occF1(occP,occR):
	return 2.0*occP*occR / (occP + occR)

#####################################################################################
#####################################################################################
# Do it

def doit(patternfile,simfunc=sc,motiffile=None,filterAnnotated=False, use_aoccs=True):
	if motiffile:
		mtable = getMotifTable(motiffile)
	else:
		mtable = getMotifTable()
	ptable = getPatternTable(patternfile, filterAnnotated)

	#mtdict = motifTableToDict_pernlbid(mtable)
	#ptdict = patternTableToDict_perpattern(ptable, use_aoccs)
	#patcoocs = getCoOccurrences(ptdict, mtdict)

	PPP = motifTableToDict_perclass_occs(mtable)
	QQQ = patternTableToDict_perpattern_occs(ptable, use_aoccs)
	S = build_S(PPP,QQQ, simfunc)
	Op,Or = build_Op_and_Or(PPP,QQQ,S,simfunc,threshold=0.75)

	#Overall measures
	print("++++++++++++++++++++++++++++++++++++++++++++")
	print("Overall")
	print("estP:  " , estPrecision(S))
	print("estR:  " , estRecall(S))
	print("estF1: " , estF1(estPrecision(S),estRecall(S)))
	print("occP:  " , occPrecision(Op))
	print("occR:  " , occRecall(Or))
	print("occF1: " , occF1(occPrecision(Op), occRecall(Or)))
	print("\n")

	# find tune families in annotated patterns
	tfs = list(set([tf[:-3] for tf in list(PPP.keys())]))
	tfs = []

	for tf in tfs:
		print("++++++++++++++++++++++++++++++++++++++++++++")
		print("Tune Family: " + tf)
		PPPtf = PPP_forTuneFamilies(PPP, [tf])
		QQQtf = QQQ_forTuneFamilies(QQQ, [tf], ptable)
		Stf = build_S(PPPtf, QQQtf, simfunc)
		Optf, Ortf = build_Op_and_Or(PPPtf, QQQtf, Stf, simfunc, threshold=0.75)
		if len(PPPtf) == 0 or len(QQQtf) == 0:
			print("estP:  N.D.")
			print("estR:  N.D.")
			print("estF1: N.D.")
			print("occP:  N.D.")
			print("occR:  N.D.")
			print("occF1: N.D.")
			print("\n")
		else:
			print("estP:  " , estPrecision(Stf))
			print("estR:  " , estRecall(Stf))
			print("estF1: " , estF1(estPrecision(Stf),estRecall(Stf)))
			print("occP:  " , occPrecision(Optf))
			print("occR:  " , occRecall(Ortf))
			print("occF1: " , occF1(occPrecision(Optf), occRecall(Ortf)))
			print("\n")

	#for annotator in list(tuneFamliesPerAnnotator.keys()):
	for annotator in []:
		print("++++++++++++++++++++++++++++++++++++++++++++")
		print("Annotator: " + annotator)
		PPPtf = PPP_forTuneFamilies(PPP, tuneFamliesPerAnnotator[annotator])
		QQQtf = QQQ_forTuneFamilies(QQQ, tuneFamliesPerAnnotator[annotator], ptable)
		Stf = build_S(PPPtf, QQQtf, simfunc)
		Optf, Ortf = build_Op_and_Or(PPPtf, QQQtf, Stf, simfunc, threshold=0.75)
		if len(PPPtf) == 0 or len(QQQtf) == 0:
			print("estP:  N.D.")
			print("estR:  N.D.")
			print("estF1: N.D.")
			print("occP:  N.D.")
			print("occR:  N.D.")
			print("occF1: N.D.")
			print("\n")
		else:
			print("estP:  " , estPrecision(Stf))
			print("estR:  " , estRecall(Stf))
			print("estF1: " , estF1(estPrecision(Stf),estRecall(Stf)))
			print("occP:  " , occPrecision(Optf))
			print("occR:  " , occRecall(Ortf))
			print("occF1: " , occF1(occPrecision(Optf), occRecall(Ortf)))
			print("\n")
