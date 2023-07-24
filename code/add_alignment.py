from Bio import Align
from Bio.Align import substitution_matrices
from pathlib import Path

aligner = Align.PairwiseAligner()
from Bio.pairwise2 import format_alignment


def do_alignment(identifier, uniprotSequence, pdbSequence, alignment_path):
    #print(f'Aligning Datapoint: {identifier}')
    if len(pdbSequence) >= 1:
        f = open(Path(alignment_path / f'{identifier}_alignment.txt'),
                 "w")
        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -11
        aligner.extend_gap_score = -1
        alignments = aligner.align(uniprotSequence, pdbSequence)
        alignments = (list(alignments))
        alignment_list = []
        for alignment in alignments:
            f.write(str(alignment))
            f.write('\n')
            f.write('\n')
            alignment = (str(alignment).strip().split('\n'))
            alignment = [''.join(['.' if m == ' ' else m for m in x]) for x in alignment]
            alignment_list.append(alignment)
    return alignment_list


def mutation_position_on_pdb(alignment_list, pos):
    which_alignment_to_go = 0
    for alignment in alignment_list:
        which_alignment_to_go += 1
        alignment_uniprot = alignment[0]
        alignment_pdb = alignment[2]
        startGap = 0
        if alignment_uniprot.startswith('.') or alignment_uniprot.startswith('-'):
            for k in alignment_uniprot:
                if k == '.' or k == '-':
                    startGap += 1
                else:
                    break
        countGap = startGap
        countResidue = 0
        canonicalRes = ' '
        pdbRes = ' '
        for j in alignment_uniprot[startGap:]:
            if j == '.' or j == '-':
                countGap += 1
            else:
                countResidue += 1
            if int(countResidue) == int(pos):
                canonicalRes = alignment_uniprot[countResidue + countGap - 1]
                try:
                    pdbRes = alignment_pdb[countResidue + countGap - 1]
                except:
                    IndexError
                    pdbRes = 'nan'
                break
        if (alignment[1][countResidue + countGap - 1] == '|') or (alignment[1][countResidue + countGap - 1] == 'X'):
            if canonicalRes == pdbRes:
                pdb_alignStatus = 'aligned'
            elif canonicalRes != pdbRes:
                pdb_alignStatus = 'aligned*'
            countGap_pdb = 0
            countResidue_pdb = 0
            pdbRes = ' '
            for j in alignment_pdb[0:countResidue + countGap - 1]:
                if j == '.' or j == '-':
                    countGap_pdb += 1
            if alignment_pdb[countResidue + countGap - 1] == '.' or alignment_pdb[
                countResidue + countGap - 1] == '-':
                mutationPositionOnPDB = 'nan'
                posPDB = 'nan'
            else:
                posPDB = countResidue + countGap - countGap_pdb
                mutationPositionOnPDB = str(posPDB)
            break
        elif (canonicalRes == pdbRes) and ((alignment[1][countResidue + countGap - 1] == '.') or (
                alignment[1][countResidue + countGap - 1] == '-')):
            pdb_alignStatus = 'not_aligned'
            mutationPositionOnPDB = 'nan'
        elif (canonicalRes != pdbRes) and ((alignment[1][countResidue + countGap - 1] == '.') or (
                alignment[1][countResidue + countGap - 1] == '-')):
            pdb_alignStatus = 'not_aligned'
            mutationPositionOnPDB = 'nan'
        elif alignment_pdb[countResidue + countGap - 1] == '.' or alignment_pdb[
            countResidue + countGap - 1] == '-':
            mutationPositionOnPDB = 'nan'
            posPDB = 'nan'
        else:
            pdb_alignStatus = 'not_aligned'
            mutationPositionOnPDB = 'nan'
    return (pdb_alignStatus, mutationPositionOnPDB, startGap, alignment_list[which_alignment_to_go - 1])


def find_position_on_pdb_for_range_annotations(posAnnotation, startGap, alignment_to_use):
    annotation_on_pdb_start = 'nan'
    annotation_on_pdb_end = 'nan'
    pos1 = int(posAnnotation.split('-')[0])
    count_gap = startGap
    count_residue = 0
    for j in alignment_to_use[0][startGap:]:
        if j == '.' or j == '-':
            count_gap += 1
        else:
            count_residue += 1
        if int(count_residue) == int(pos1):  # count gaps until the first position
            break
    annotation_on_up_start = int(pos1) + int(count_gap)

    pos2 = int(posAnnotation.split('-')[1])
    count_gap = startGap
    count_residue = 0
    for j in alignment_to_use[0][startGap:]:
        if j == '.' or j == '-':
            count_gap += 1
        else:
            count_residue += 1
        if int(count_residue) == int(pos2):  # count gaps until the first position
            break

    annotation_on_up_end = int(pos2) + int(count_gap)
    try:
        pdb_residue_start = alignment_to_use[2][annotation_on_up_start - 1].strip()
        if (pdb_residue_start == '.') or (pdb_residue_start == '-'):
            for ran in range(len(alignment_to_use[2][(annotation_on_up_start - 1):annotation_on_up_end])):
                if (alignment_to_use[2][(annotation_on_up_start - 1):annotation_on_up_end][ran] != '.') and \
                        (alignment_to_use[2][(annotation_on_up_start - 1):annotation_on_up_end][ran] != '-') and \
                        ((alignment_to_use[1][(annotation_on_up_start - 1):annotation_on_up_end][ran] == '|') or
                         (alignment_to_use[1][(annotation_on_up_start - 1):annotation_on_up_end][ran] == 'X')):
                    annotation_on_up_start += ran
                    break
        elif (pdb_residue_start != '.') and (pdb_residue_start != '-') and \
                ((alignment_to_use[1][annotation_on_up_start - 1] == '.') or (
                        alignment_to_use[1][annotation_on_up_start - 1] == '-')):
            for ran in range(len(alignment_to_use[2][(annotation_on_up_start - 1):annotation_on_up_end])):
                if ((alignment_to_use[1][(annotation_on_up_start - 1):annotation_on_up_end][ran] == '|') or
                        (alignment_to_use[1][(annotation_on_up_start - 1):annotation_on_up_end][ran] == 'X')):
                    annotation_on_up_start += ran
                    break
        count_gap_pdb = 0
        if annotation_on_up_start != 'nan':
            for q in alignment_to_use[2][0:annotation_on_up_start - 1]:
                if q == '.' or q == '-':
                    count_gap_pdb += 1
            if alignment_to_use[1][annotation_on_up_start] == '-' or alignment_to_use[1][annotation_on_up_start] == '.':
                annotation_on_pdb_start = 'nan'
            else:
                annotation_on_pdb_start = int(annotation_on_up_start) - count_gap_pdb
        else:
            annotation_on_pdb_start = 'nan'
    except:
        IndexError
    try:
        pdb_residue_end = alignment_to_use[2][annotation_on_up_end - 1].strip()
        if pdb_residue_end == '.' or pdb_residue_end == '-':
            for ran in range(len(alignment_to_use[2][(annotation_on_up_start - 1):annotation_on_up_end])):
                if ((alignment_to_use[1][annotation_on_up_start - 1:annotation_on_up_end][ran] == '.') or
                        (alignment_to_use[1][(annotation_on_up_start - 1):][ran] == '-')):
                    annotation_on_up_start += (ran - 1)
                    annotation_on_up_end = annotation_on_up_start
                    break
        elif (pdb_residue_end != '.') and (pdb_residue_end != '-') and \
                ((alignment_to_use[1][annotation_on_up_end - 1] == '.') or (
                        alignment_to_use[1][annotation_on_up_end - 1] == '-')):
            for ran in range(len(alignment_to_use[2][(annotation_on_up_start - 1):annotation_on_up_end])):
                if ((alignment_to_use[1][annotation_on_up_start - 1:annotation_on_up_end][ran] == '.') or
                        (alignment_to_use[1][(annotation_on_up_start - 1):][ran] == '-')):
                    annotation_on_up_start += (ran - 1)
                    annotation_on_up_end = annotation_on_up_start
                    break
        count_gap_pdb = 0
        if annotation_on_up_end != 'nan':
            for q in alignment_to_use[2][0:annotation_on_up_end - 1]:
                if q == '.' or q == '-':
                    count_gap_pdb += 1
            if alignment_to_use[1][annotation_on_up_end - 1] == '-' or alignment_to_use[1][
                annotation_on_up_end - 1] == '.' and annotation_on_pdb_start == 'nan':
                annotation_on_pdb_end = 'nan'
            elif alignment_to_use[1][annotation_on_up_end - 1] == '-' or alignment_to_use[1][
                annotation_on_up_end - 1] == '.' and annotation_on_pdb_start != 'nan':
                annotation_on_pdb_end = int(annotation_on_up_end) - count_gap_pdb
            else:
                annotation_on_pdb_end = int(annotation_on_up_end) - count_gap_pdb
        else:
            annotation_on_pdb_end = 'nan'
    except:
        IndexError  # Say isoform 2 is matched with the length 100, but canonical is 150 aa long. If there is an annotation at 105. position, for the isoform it throws an index error.

    if annotation_on_pdb_start == 'nan' and annotation_on_pdb_end != 'nan':
        annotation_on_pdb_start = annotation_on_up_start - count_gap_pdb
    if annotation_on_pdb_start == annotation_on_pdb_end:
        annotation_on_pdb_start = 'nan'
        annotation_on_pdb_end = 'nan'
    return annotation_on_up_start, annotation_on_up_end, annotation_on_pdb_start, annotation_on_pdb_end


def annotation_pos_on_pdb(annot_positions, startGap, alignment_to_use, identifier):
    newpos = []
    if annot_positions != 'nan':
        annot_positions = (str(annot_positions).replace("'", ''))
        annot_positions = (str(annot_positions).replace('[', ''))
        annot_positions = (str(annot_positions).replace("]", ''))
        positionList_perAnnotation = annot_positions.split(',')
        positionList_perAnnotation = [h.strip() for h in positionList_perAnnotation]

        position_start_on_pdb = 'nan'
        position_end_on_pdb = 'nan'
        try:
            positionList_perAnnotation = [i for i in positionList_perAnnotation if i != 'nan']
        except:
            TypeError
        for position in range(len(positionList_perAnnotation)):
            if ('-' not in str(positionList_perAnnotation[position])) and (str(positionList_perAnnotation[position]) != '?') and (str(positionList_perAnnotation[position]) != '') and (len(str(positionList_perAnnotation[position])) != 0):
                count_gap = startGap
                count_residue = 0
                for j in alignment_to_use[0][startGap:]:
                    if j == '.' or j == '-':
                        count_gap += 1
                    else:
                        count_residue += 1
                    try:
                        if int(count_residue) == int(positionList_perAnnotation[position]):
                            break
                    except:
                        ValueError

                annotation_on_up = int(positionList_perAnnotation[position]) + int(count_gap)
                try:
                    pdb_residue_start = alignment_to_use[2][annotation_on_up - 1].strip()
                except:
                    IndexError
                    pdb_residue_start = 'nan'
                if pdb_residue_start != 'nan':
                    try:
                        if (pdb_residue_start == '.') or (pdb_residue_start == '-'):
                            for ran in range(len(alignment_to_use[2][(annotation_on_up - 1):annotation_on_up])):
                                if (alignment_to_use[2][(annotation_on_up - 1):annotation_on_up][ran] != '.') and \
                                        (alignment_to_use[2][(annotation_on_up - 1):annotation_on_up][
                                             ran] != '-') and \
                                        ((alignment_to_use[1][(annotation_on_up - 1):annotation_on_up][
                                              ran] == '|') or
                                         (alignment_to_use[1][(annotation_on_up - 1):annotation_on_up][
                                              ran] == 'X')):
                                    annotation_on_up += ran
                                    break
                        elif (pdb_residue_start != '.') and (pdb_residue_start != '-') and \
                                ((alignment_to_use[1][annotation_on_up - 1] == '.') or (
                                        alignment_to_use[1][annotation_on_up - 1] == '-')):
                            for ran in range(len(alignment_to_use[2][(annotation_on_up - 1):annotation_on_up])):
                                if ((alignment_to_use[1][(annotation_on_up - 1):annotation_on_up][ran] == '|') or
                                        (alignment_to_use[1][(annotation_on_up - 1):annotation_on_up][ran] == 'X')):
                                    annotation_on_up += ran
                                    break
                        count_gap_pdb = 0
                        for q in alignment_to_use[2][0:annotation_on_up - 1]:
                            if q == '.' or q == '-':
                                count_gap_pdb += 1
                        if alignment_to_use[1][annotation_on_up] == '-' or alignment_to_use[1][
                            annotation_on_up] == '.':
                            annotation_on_pdb = 'nan'
                        else:
                            annotation_on_pdb = int(annotation_on_up) - count_gap_pdb

                        if count_gap_pdb == annotation_on_up:
                            annotation_on_pdb = 'nan'
                        try:
                            if alignment_to_use[2][count_gap_pdb + annotation_on_pdb - 1] == '.' or alignment_to_use[2][
                                count_gap_pdb + annotation_on_pdb - 1] == '-':
                                annotation_on_pdb = 'nan'
                        except:
                            IndexError
                            annotation_on_pdb = 'nan'
                    except:
                        IndexError
                        annotation_on_pdb = 'nan'

                    newpos.append(annotation_on_pdb)

            elif ('-' in str(positionList_perAnnotation[position])) and (
                    str(positionList_perAnnotation[position]) != '?') and (
                    str(positionList_perAnnotation[position]) != ' ') and (
                    len(str(positionList_perAnnotation[position])) != 0):
                try:
                    position_start_on_pdb = \
                        find_position_on_pdb_for_range_annotations(positionList_perAnnotation[position],
                                                                   startGap, alignment_to_use)[2]
                    position_end_on_pdb = \
                        find_position_on_pdb_for_range_annotations(positionList_perAnnotation[position],
                                                                   startGap, alignment_to_use)[3]
                except:
                    ValueError
                newpositions = str(position_start_on_pdb) + '-' + str(position_end_on_pdb)
                newpos.append(newpositions)
            else:
                pass
    try:
        newpos = [i for i in newpos if i != 'nan']
    except:
        TypeError
    return newpos


def final_stage(df, annotation_list, alignment_path):
    for i in df.index:
        identifier = df.at[i, 'uniprotID'] + '_' + df.at[i, 'pdbID'] + '_' + df.at[i, 'chain'] + '_'
        alignment_list = do_alignment(identifier, df.at[i, 'uniprotSequence'], df.at[i, 'pdbSequence'], alignment_path)
        df.at[i, 'pdb_alignStatus'] = mutation_position_on_pdb(alignment_list, df.at[i, 'pos'])[0]
        df.at[i, 'mutationPositionOnPDB'] = mutation_position_on_pdb(alignment_list, df.at[i, 'pos'])[1]
        startGap = mutation_position_on_pdb(alignment_list, df.at[i, 'pos'])[2]
        alignment_to_use = mutation_position_on_pdb(alignment_list, df.at[i, 'pos'])[3]
        for annot in annotation_list:
            df.at[i, annot] = annotation_pos_on_pdb(df.at[i, annot], startGap, alignment_to_use, identifier)
        if str(df.at[i, 'domStart']) != 'nan' and str(df.at[i, 'domEnd']) != 'nan' and \
                ((str(df.at[i, 'domStart']) != '-1' and str(df.at[i, 'domEnd']) != '-1' and
                  str(df.at[i, 'domStart']) != '-1.0' and str(df.at[i, 'domEnd']) != '-1.0')):
            domainLoc = str(df.at[i, 'domStart']).split('.')[0] + '-' + str(df.at[i, 'domEnd']).split('.')[0]
            domain_pos = find_position_on_pdb_for_range_annotations(domainLoc, startGap, alignment_to_use)
            df.at[i, 'domainStartonPDB'] = domain_pos[2]
            df.at[i, 'domainEndonPDB'] = domain_pos[3]
        elif str(df.at[i, 'domStart']) != '-1' or str(df.at[i, 'domEnd']) != '-1' or \
                str(df.at[i, 'domStart']) != '-1.0' or str(df.at[i, 'domEnd']) != '-1.0':
            df.at[i, 'domainStartonPDB'] = 'nan'
            df.at[i, 'domainEndonPDB'] = 'nan'
    return df

def alignment(dataframe_to_align, annotation_list, alignment_path):
    domainList = ['domStart', 'domEnd']
    result = final_stage(dataframe_to_align, annotation_list, alignment_path)
    return result
