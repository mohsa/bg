import json
import os
import webbrowser
import numpy as np
import itertools
import cv2
import collections

from constants import *
from utility import *

class HomographyGenerator:
    verbose = False

    def __init__(self, annotations_filename='../Data/annotations.json', verbose=False):
        with open(annotations_filename) as annotations_file:
            annotation = json.load(annotations_file)
            self.annotations = collections.OrderedDict(sorted(annotation.items(), key=lambda t: t[0]))

    def get_lines(self, filename):
        lines = self.annotations[filename]

        yard_lines = lines['yard_lines']
        hash_lines = lines['hash_lines']

        yard_lines = sorted(yard_lines)
        hash_lines = sorted(hash_lines, key=lambda line: line[0][1])

        # Because images are downscaled by 2
        yard_lines = [list(np.array(line) / 2) for line in yard_lines]
        hash_lines = [list(np.array(line) / 2) for line in hash_lines]

        return yard_lines, hash_lines


    def get_homography(self, filename, refFrameFile):
        # 1. Get lines
        yard_lines, hash_lines = self.get_lines(filename)
        yard_lines_ref, hash_lines_ref = self.get_lines(refFrameFile)

        if len(yard_lines) < 2:
            raise ValueError('Too few yard lines.')

        if len(hash_lines) < 2:
            raise ValueError('Too few hash lines.')
        
        if len(yard_lines_ref) < 2:
            raise ValueError('Too few yard lines.')

        if len(hash_lines_ref) < 2:
            raise ValueError('Too few hash lines.')


        # 2. Calculate intersections
        target_points = np.array(line_intersections(yard_lines, hash_lines))
        reference_points = np.array(line_intersections(yard_lines_ref, hash_lines_ref))

        if len(target_points) < 4:
            raise ValueError('Too few target points of intersection.')

        if len(reference_points) < 4:
            raise ValueError('Too few target points of intersection.')


        # target_points, reference_points. dx, dy =  pair_points(target_points, reference_points)

        #league = filename_to_league(filename)
        reference_points = reference_points[:len(target_points)]

        if len(target_points) > len(reference_points):
            raise ValueError('Too many target points.')

        # 3. Find homography between reference and target points.
        h, _ = cv2.findHomography(target_points, reference_points)
        return h

    def get_all_homographies(self):
        homographies = {}
        for filename in self.annotations.keys():
            try:
                h = self.get_homography(filename)
            except ValueError:
                continue

            homographies[filename] = h

        return homographies

    def get_ground_truth(self, output_format='corner', only_nfl=False):
        output = {}

        total, errors = 0, 0
        error_messages = {}

        refFrameFile = list(self.annotations.keys())[0] #added first frame as reference frame


        for filename in self.annotations.keys():
            if only_nfl and filename_to_league(filename) == 'NFL':
                continue

            total += 1
            
            try:
                h = self.get_homography(filename, refFrameFile)
                
            except ValueError as e:
                errors += 1
                e = str(e)
                refFrameFile = filename # if error go .... next frame
                if e in error_messages:
                    error_messages[e] += 1
                else:
                    error_messages[e] = 1
                continue
            
            output[filename] = {}             
 
            if output_format == 'corner':
                new_corners = [np.dot(h, (x, y, 1)) for x, y in ORIGINAL_CORNERS]
                new_corners = np.array([(x/w, y/w) for (x, y, w) in new_corners])
                new_corners -= np.array(ORIGINAL_CORNERS)
                output[filename][refFrameFile] = new_corners.tolist()
                refFrameFile = filename
            elif output_format == 'homography':
                output[filename][refFrameFile] = h.tolist()
                refFrameFile = filename 
            else:
               raise ValueError('Invalid output format.')
            
        if self.verbose:
            print("Successful exports: {0} Erroneous exports: {1} Success rate: {2}".format(total - errors, errors, 1 - float(errors) / (total)))
            print("Type of errors")
            for e, v in error_messages.items():
                print("{0} | {1}".format(e, v))

        return output


    def export_all_transformed_corners(self, output_filename, output_format='corner', only_nfl=False):
        output = get_ground_truth(output_format, only_nfl)
        with open(output_filename, 'w') as outfile:
            json.dump(output, outfile)

if __name__ == "__main__":
    gen = HomographyGenerator()
    gen.export_all_transformed_corners('homographies.json', 'corner', True)
