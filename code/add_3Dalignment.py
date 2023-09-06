"""
This code file produces alignments between the structure and the sequence for a given protein.

"""

import math
import glob
import numpy as np
from Bio import Align
import gzip
from pathlib import Path
from Bio.Align import substitution_matrices
aligner = Align.PairwiseAligner()
def convert_non_standard_amino_acids(sequence):
    """
    Convert non-standard or ambiguous amino acid codes to their closest relatives.
    """

    # Define a dictionary to map non-standard codes to standard amino acids
    conversion_dict = {
        'B': 'D',  # Aspartic Acid (D) is often used for B (Asx)
        'Z': 'E',  # Glutamic Acid (E) is often used for Z (Glx)
        'X': 'A',  # Alanine (A) is a common placeholder for unknown/ambiguous
        'U': 'C',  # Cysteine (C) is often used for Selenocysteine (U)
        'J': 'L',  # Leucine (L) is often used for J (Leu/Ile)
        'O': 'K',  # Lysine (K) is often used for O (Pyrrolysine)
        # '*' or 'Stop' represents a stop codon; you may replace with '' to remove
        '*': '',
    }

    # Replace non-standard codes with their closest relatives
    converted_sequence = ''.join([conversion_dict.get(aa, aa) for aa in sequence])

    return converted_sequence
def distance(x1, y1, z1, x2, y2, z2):
    d = math.sqrt(math.pow(x2 - x1, 2) +
                  math.pow(y2 - y1, 2) +
                  math.pow(z2 - z1, 2) * 1.0)
    return d


def find_distance(coordMut, coordAnnot):
    if coordMut != np.NaN:
        try:
            dist = distance(float(coordMut[0]), float(coordMut[1]), float(coordMut[2]), float(coordAnnot[0]),
                            float(coordAnnot[1]), float(coordAnnot[2]))
            return "%.2f" % dist
        except:
            ValueError
            dist = 'nan'
            return dist
    else:
        return np.NaN


def threeToOne(variant):
    if variant == "ALA":
        variant = "A"
    elif variant == "ARG":
        variant = "R"
    elif variant == "VAL":
        variant = "V"
    elif variant == "GLU":
        variant = "E"
    elif variant == "PRO":
        variant = "P"
    elif variant == "LEU":
        variant = "L"
    elif variant == "GLY":
        variant = "G"
    elif variant == "ASN":
        variant = "N"
    elif variant == "SER":
        variant = "S"
    elif variant == "GLN":
        variant = "Q"
    elif variant == "THR":
        variant = "T"
    elif variant == "MET":
        variant = "M"
    elif variant == "LYS":
        variant = "K"
    elif variant == "ASP":
        variant = "D"
    elif variant == "ILE":
        variant = "I"
    elif variant == "PHE":
        variant = "F"
    elif variant == "TRP":
        variant = "W"
    elif variant == "TYR":
        variant = "Y"
    elif variant == "HIS":
        variant = "H"
    elif variant == "CYS":
        variant = "C"
    elif variant == 'UNK':
        variant = 'X'
    elif variant == 'ASX':
        variant = 'O'
    return (variant)


def get_coords(annot, alignments, coords, resnums_for_sasa, mode):
    if mode == 1:
        for alignment in alignments[0]:
            alignment = (str(alignment).strip().split('\n'))
            startGap = 0
            if alignment[0].startswith('.'):
                for k in alignment[0]:
                    if k == '.' or k == '-':
                        startGap += 1
                    else:
                        break
            countGap = startGap
            countResidue = 0
            for j in alignment[0][startGap:]:
                if j == '.' or j == '-':
                    countGap += 1
                else:
                    countResidue += 1
                if countResidue == float(annot):
                    break
            countGap_pdb = 0
            countResidue_pdb = 0
            for m in alignment[2][0:countResidue + countGap - 1]:
                if m == '.' or m == '-':
                    countGap_pdb += 1
            posAtom = countResidue + countGap - countGap_pdb

            realpdbStart = 0
            for j in alignment[2]:
                if j == '.' or j == '-':
                    realpdbStart += 1
                else:
                    break

            if (alignment[2][countResidue + countGap - 1] != '-') and (float(annot) >= float(realpdbStart) + 1):
                try:
                    coordinates = alignments[1]
                    residue_numbers = alignments[2]
                    coordWeWant = coordinates[posAtom - 1]
                    residue_number_we_want = residue_numbers[posAtom - 1]

                except:
                    IndexError
                    coordWeWant = 'nan'
            else:
                coordWeWant = 'nan'
            return coordWeWant, posAtom, residue_number_we_want
    if mode == 2:
        if annot != 'nan':
            if int(annot) <= 1400:
                alignment = (str(alignments).strip().split('\n'))
                startGap = 0
                if alignment[0].startswith('.'):
                    for k in alignment[0]:
                        if k == '.' or k == '-':
                            startGap += 1
                        else:
                            break
                countGap = startGap
                countResidue = 0
                for j in alignment[0][startGap:]:
                    if j == '.' or j == '-':
                        countGap += 1
                    else:
                        countResidue += 1
                    if countResidue == float(annot):
                        break
                countGap_pdb = 0
                countResidue_pdb = 0
                for m in alignment[2][0:countResidue + countGap - 1]:
                    if m == '.' or m == '-':
                        countGap_pdb += 1
                posAtom = countResidue + countGap - countGap_pdb
                realpdbStart = 0
                for j in alignment[2]:
                    if j == '.' or j == '-':
                        realpdbStart += 1
                    else:
                        break
                if len(alignment[2]) > (countResidue + countGap - 1):
                    if (alignment[2][countResidue + countGap - 1] != '-') and (float(annot) >= float(realpdbStart) + 1):
                        try:
                            coordinates = coords
                            residue_numbers = resnums_for_sasa
                            coordWeWant = coordinates[posAtom - 1]
                            residue_number_we_want = residue_numbers[posAtom - 1]
                        except:
                            IndexError
                            coordWeWant = 'nan'
                            residue_number_we_want = 'nan'
                    else:
                        coordWeWant = 'nan'
                        residue_number_we_want = 'nan'
                    return coordWeWant, posAtom, residue_number_we_want
                else:
                    coordWeWant = 'nan'
                    residue_number_we_want = 'nan'
                    return coordWeWant, posAtom, residue_number_we_want
            else:
                return np.NaN, np.NaN, np.NaN
        else:
            return np.NaN, np.NaN, np.NaN


def get_alignments_3D(identifier, model_num, pdb_path, pdbSequence, source, chain, pdbID, mode, path_3D_alignment,file_format = 'gzip'):
    pdbSequence = convert_non_standard_amino_acids(pdbSequence)
    if mode == 1:
        atomSequence = ''
        coords = []
        resnums_for_sasa = []
        with open(pdb_path, encoding="utf8") as f:
            for line in f.readlines():
                if source != 'MODBASE':
                    if line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA' and line[21].upper() == chain.upper():
                        atomSequence += threeToOne(line[17:20].strip())
                        coords.append([line[31:38].strip(), line[39:46].strip(), line[47:54].strip()])
                        resnums_for_sasa.append(line[22:26].strip())
                    elif line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA' and line[21] == ' ':
                        atomSequence += threeToOne(line[17:20].strip())
                        coords.append([line[31:38].strip(), line[39:46].strip(), line[47:54].strip()])
                        resnums_for_sasa.append(line[22:26].strip())
                else:
                    if line[0:7].strip() == 'ATOM' and line[13:15].strip() == 'CA':
                        atomSequence += threeToOne(line[17:20].strip())
                        coords.append([line[31:38].strip(), line[39:46].strip(), line[47:54].strip()])
                        resnums_for_sasa.append(line[22:26].strip())

        f = open(Path(path_3D_alignment / f'{identifier}_{pdbID}_{str(chain)}_alignment.txt'),"w")
        atomSequence = convert_non_standard_amino_acids(atomSequence)

        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1
        alignments = aligner.align(pdbSequence, atomSequence)
        alignments = (list(alignments))
        for alignment in alignments:
            f.write(str(alignment))
            f.write('\n')
            f.write('\n')
        return alignments, coords, resnums_for_sasa
    elif mode==2:
            atomSequence = ''
            coords = []
            resnums_for_sasa = []
            if file_format == 'txt':
                with open(name, encoding="utf8") as f:
                    for line in f.readlines():
                        if line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA':
                            atomSequence += threeToOne(line[17:20].strip())
                            coords.append([line[31:38].strip(), line[39:46].strip(), line[47:54].strip()])
                            resnums_for_sasa.append(line[22:26].strip())
                        elif line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA' and line[21] == ' ':
                            atomSequence += threeToOne(line[17:20].strip())
                            coords.append([line[31:38].strip(), line[39:46].strip(), line[47:54].strip()])
                            resnums_for_sasa.append(line[22:26].strip())
            elif file_format == 'gzip':
                with gzip.open(pdb_path, mode='rb') as f:
                    for line in f:
                        line = line.decode()
                        if line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA':
                            atomSequence += threeToOne(line[17:20].strip())
                            coords.append([line[31:38].strip(), line[39:46].strip(), line[47:54].strip()])
                            resnums_for_sasa.append(line[22:26].strip())
                        elif line[0:4].strip() == 'ATOM' and line[13:15].strip() == 'CA' and line[21] == ' ':
                            atomSequence += threeToOne(line[17:20].strip())
                            coords.append([line[31:38].strip(), line[39:46].strip(), line[47:54].strip()])
                            resnums_for_sasa.append(line[22:26].strip())
            f = open(Path(path_3D_alignment / f'{identifier}_{str(model_num)}_3Dalignment.txt'),"w")
            aligner.mode = 'local'
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
            aligner.open_gap_score = -11
            aligner.extend_gap_score = -1
            atomSequence = convert_non_standard_amino_acids(atomSequence)
            alignments = aligner.align(pdbSequence, atomSequence)
            alignments = (list(alignments))
            for alignment in alignments:
                f.write(str(alignment))
                f.write('\n')
                f.write('\n')
            return alignments, coords, resnums_for_sasa
