from turtle import right
from pkg_resources import yield_lines
import cv2
import numpy as np
from src.util import last_arg, to_images

# Constants.
kRatio = 3
kGap = 2

# A point.
class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

class Segment:
  def __init__(self, A, B):
    self.A, self.B = A, B

  # Return true if `this` segment intersects with `other` segment.
  def intersects(self, other):
    def ccw(A,B,C):
      return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
    return ccw(self.A, other.A, other.B) != ccw(self.B, other.A, other.B) and ccw(self.A, self.B, other.A) != ccw(self.A, self.B, other.B)

# A connected component.
class ConnectedComponent:
  def __init__(self, x, y, w, h, a):
    self.x, self.y, self.w, self.h, self.a = x, y, w, h, a

  def __str__(self):
    return f'x={self.x}, y={self.y}, w={self.w}, h={self.h}'

class Sheet:
  # A page.
  class Page:
    # Baseline:
    class Baseline:
      class Lyrics:
        def __init__(self, master):
          self.master = master

        def set_coordinate(self, y):
          self.y = y

        # def fetch_initial_text(self):
        #   for i, cc in enumerate(self.master.master.ccs):

        
        # TODO: avoid duplicate code -> use inheritance!

      # Baseline constructor.
      def __init__(self, master, index, y):
        self.master = master
        self.index = index
        self.y = y
        self.lyrics = self.Lyrics(self)

      def __str__(self):
        return f'y = {self.y}'

      @staticmethod
      def distance(n1, n2):
        # Is `n2` below `n1`?
        if n1.y < n2.y:
          return min(abs(n1.y - n2.y), abs(n1.y + n1.h - n2.y))
        else:
          return min(abs(n2.y - n1.y), abs(n2.y + n2.h - n1.y))

      @staticmethod
      def intersection(s1, s2):
        return max(s1[0], s2[0]) <= min(s1[1], s2[1])

      def fetch_touching_neumes(self):        
        def touches(cc):
          s1 = (self.y - self.master.master.oligon_height, self.y + self.master.master.oligon_height)
          s2 = (cc.y, cc.y + cc.h)
          return self.intersection(s1, s2)

        # Determine neumes touching the baseline.
        self.touching_neumes = []
        for i, cc in enumerate(self.master.ccs):
          if touches(cc):
            self.touching_neumes.append(i)

      def fetch_final_neumes(self):
        prev, next = None, None
        if self.index:
          prev = self.master.neumes_baselines[self.index - 1]
          print(f'prev={prev}')
        if self.index + 1 != len(self.master.neumes_baselines):
          next = self.master.neumes_baselines[self.index + 1]
          print(f'next={next}')

        def projection_intersection(n1, n2):
          s1, s2 = (n1.x, n1.x + n1.w), (n2.x, n2.x + n2.w)
          return self.intersection(s1, s2)

        def is_below_baseline(cc):
          return self.y < cc.y + cc.h

        # Check if `n2` is below `n1`.
        def is_below_neume(n1, n2):
          return (n2.y > n1.y) and projection_intersection(n1, n2)

        def is_above_baseline(cc):
          return cc.y < self.y

        # Check if `n2` is above `n1`.
        def is_above_neume(n1, n2):
          return (n2.y < n1.y) and projection_intersection(n1, n2)

        def fetch(i):
          return self.master.ccs[i]

        prev_y, next_y = 0, self.master.height
        if prev is not None:
          prev_y = prev.y
        if next is not None:
          next_y = next.y

        # Second iteration
        self.suspended_neumes = []
        for i, cc in enumerate(self.master.ccs):
          # Already taken by us?
          if i in self.touching_neumes:
            continue

          # Already taken by `prev`?
          if prev is not None and (i in prev.touching_neumes):
            continue

          # Already taken by `next`?
          if next is not None and (i in next.touching_neumes):
            continue

          # Not in our range?
          if (cc.y < prev_y) or (next_y < cc.y):
            continue

          # Is it below us?
          if is_below_baseline(cc):
            # print(f'cc={cc} is below')
            for j in self.touching_neumes:
              if is_below_neume(fetch(j), cc) and (self.distance(fetch(j), cc) < kGap * self.master.master.oligon_height):
                self.suspended_neumes.append(i)
                break

          # Is it above us?
          if is_above_baseline(cc):
            # print(f'cc={cc} is above')
            for j in self.touching_neumes:
              if is_above_neume(fetch(j), cc) and (self.distance(fetch(j), cc) < kGap * self.master.master.oligon_height):
                self.suspended_neumes.append(i)
                break

        # Build the final neumes list.
        self.neumes = self.touching_neumes + self.suspended_neumes

    # Constructor.
    def __init__(self, master, image):
      self.master = master
      self.image = image
      gray = cv2.cvtColor(np.array(image.convert('RGB'))[:, :, ::-1].copy(), cv2.COLOR_BGR2GRAY)
      self.height, self.width = gray.shape

      self.thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      output = cv2.connectedComponentsWithStats(self.thresh, 8, cv2.CV_32S)
      (_, _, tmp, self.centroids) = output
      self.ccs = []
      for (x, y, w, h, a) in tmp:
        self.ccs.append(ConnectedComponent(x, y, w, h, a))

    # Compute the horizontal projection of connected components with `w / h` < `ratio`. 
    def compute_horizontal_projection(self, ratio=kRatio):
      hs = np.zeros(self.master.shape[1] + 1)

      def collect_horizontal_runlengths(cc):
        for y_ in range(cc.y, cc.y + cc.h):
          hs[y_] += np.count_nonzero(self.thresh[y_][cc.x : cc.x + cc.w] == 255)

      for cc in self.ccs:
        if cc.w / cc.h < ratio:
          continue
        collect_horizontal_runlengths(cc)
      hs /= self.master.oligon_width
      return hs

    # Get min peaks.
    @staticmethod
    def get_max_peaks(xs):
      from scipy.signal import find_peaks
      peaks, _ = find_peaks(xs, height=0)
      return peaks
    
    # Get max peaks.
    @staticmethod
    def get_min_peaks(xs):
      from scipy.signal import find_peaks
      peaks, _ = find_peaks(-xs)
      return peaks

    # Compute baselines.
    def compute_neumes_baselines(self, theta=0.8):
      hs = self.compute_horizontal_projection()
      peaks = self.get_max_peaks(hs)
      
      # Extract only peaks which correspond to baselines.
      # TODO: this is not the method specified in the paper
      # TODO: we should first take the maximum within an interval of oligon_width.
      peaks = peaks[np.where(hs[peaks] > theta)]

      # for peak in peaks:
      #   print(f'remain={peaks[(peak <= peaks) & (peaks < peak + self.master.oligon_width)]}')
      #   window = peaks[(peak <= peaks) & (peaks < peak + self.master.oligon_width)]
      #   arg = window[np.argmax(hs[window])]
      #   print(f'arg={arg}')

      new_peaks = []
      index = 0
      while index < len(peaks):
        ptr = index + 1
        argmax = index
        while ptr < len(peaks) and peaks[ptr] - peaks[index] <= self.master.oligon_width:
          if hs[peaks[ptr]] > hs[peaks[argmax]]:
            argmax = ptr
          ptr += 1
        new_peaks.append(peaks[argmax])
        index = ptr

      # Reigster all baselines.
      self.neumes_baselines = []
      for index, nb in enumerate(new_peaks):
        self.neumes_baselines.append(self.Baseline(self, index, nb))
      return

    # Plot the baselines.
    def plot_neumes_baselines(self):
      import matplotlib.pyplot as plt
      import matplotlib.patches as patches

      # Create figure and axes
      fig, ax = plt.subplots(figsize=(self.height / 10, self.width / 10))

      # Display the image
      ax.imshow(self.image)

      self.compute_neumes_baselines()
      for index, nb in enumerate(self.neumes_baselines):
        rect = patches.Rectangle((0, max(0, nb.y - self.master.oligon_height / 2)), self.master.shape[0], self.master.oligon_height, linewidth=2.5, edgecolor='purple', facecolor='none', label=f'{index}')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        ax.annotate(f'{index}', (cx, cy), color='green', weight='bold', fontsize=16, ha='center', va='center')
      plt.show()

    # Compute the horizontal projetion, by considering all connected componets.
    def compute_raw_horizontal_projection(self):
      return np.sum(self.thresh, axis=1) / 255
      
    # Plot the horizontal projetion, by considering all connected componets.
    def plot_raw_horizontal_projection(self):
      import matplotlib.pyplot as plt
      f = plt.figure()
      f.set_figwidth(50)
      f.set_figheight(10)

      hs = self.compute_raw_horizontal_projection()
      self.compute_neumes_baselines()

      for index in range(len(self.neumes_baselines)):
        y = self.neumes_baselines[index].y
        plt.plot([y], [hs[y]], marker='o', markersize=15, color="red")
        if index:
          mid = (self.neumes_baselines[index].y + self.neumes_baselines[index - 1].y) / 2
          plt.axvline(x = mid)
      plt.plot(hs, color='black')

    # TODO: this is not so clean. We shouldn't record the neumes baselines in a function with a different name.
    def compute_full_baselines(self):
      # First compute the neumes baselines.
      self.compute_neumes_baselines()

      # Compute the raw horizontal projection.
      rhp = self.compute_raw_horizontal_projection()

      # print([str(nb) for nb in self.neumes_baselines])

      # def interpolate(b1, b2):
      #   print(f'b1={str(b1)}')
      #   print(f'b2={str(b2)}')
      #   assert b1.y < b2.y
      #   fst_pos = b2.y - np.argmax(rhp[b1.y : b2.y][::-1] == 0)

      #   # TODO: take the one closest to the center.
      #   mid = b1.y + (b2.y - b1.y) / 2
      #   print(f'fst_pos={fst_pos} mid={mid}')
      #   assert fst_pos >= mid

      #   b2.set_lower_bound(fst_pos)
      #   b1.set_upper_bound(fst_pos)

      #   while fst_pos >= b1.y and rhp[fst_pos] == 0:
      #     fst_pos -= 1
      #   fst_pos += 1

      #   # TODO: take the smallest min before the greatest max (which shouldn't be the baseline itself)
      #   # For that, make sure that we take a local maximum, which is *at least* oligon_height apart from us.

      #   # TODO: should we start directly with `b1.y + self.master.oligon_height` and then find the maxs?
      #   # If so, pay attention to also add `b1.y + self.master.oligon_height`
      #   max_peaks = b1.y + self.get_max_peaks(rhp[b1.y : fst_pos])
      #   max_peaks = max_peaks[max_peaks > b1.y + self.master.oligon_height]
        
      #   print(f'! max={max_peaks}')
      #   print(f'! rhp_max={rhp[max_peaks]}')
        
      #   rightmost_max_index = max_peaks[last_arg(max_peaks, np.argmax)]

      #   # rightmost_min_before_max_index = min_peaks
      #   print(f'rightmost_max_index={rightmost_max_index}')

      #   safe_start_position = b1.y + self.master.oligon_height
      #   rightmost_min_index = safe_start_position + last_arg(rhp[safe_start_position : rightmost_max_index], np.argmin)

      #   print(f'rm_min_index={rightmost_min_index}')

      #   b1.lyrics.set_lower_bound(rightmost_min_index)

      def find_lyrics(b1, b2):
        mid = b1.y + (b2.y - b1.y) / 2
        max_peaks = b1.y + self.get_max_peaks(rhp[b1.y : b2.y])
        mask = (b1.y + self.master.oligon_height <= max_peaks) & (max_peaks <= mid)
        max_peaks = max_peaks[mask]

        # print(f'b1.y={b1.y}, b2.y={b2.y}, mid={mid}, max_peaks={max_peaks}, values={rhp[max_peaks]}')
        ind = np.argpartition(rhp[max_peaks], -2)[-2:]
        # print(f'ind={ind}')

        # TODO: what if `len(ind) == 1`?
        assert len(ind) == 2
        max1, max2 = max_peaks[ind]

        # print(f'max1={max1}, max2={max2}')
        b1.lyrics.set_coordinate(max1 + (max2 - max1) / 2)
        
      # TODO: could go wrong, when the end of sentence is really short!
      # TODO: and we also have page numbers.
      self.neumes_baselines.append(self.Baseline(self, 0, self.height))
      for i in range(1, len(self.neumes_baselines)):
        find_lyrics(self.neumes_baselines[i - 1], self.neumes_baselines[i])
      self.neumes_baselines.pop()
      # TODO: infer from other pages, what the distance to the lyrics is.
      # TODO: or apply other heuristic, e.g., infer the center and get the 2 maximums
      # self.neumes_baselines[-1].lyrics.set_coordinate(self.height)
      # # TODO: we could do better here.
      # self.neumes_baselines[0].set_lower_bound(0)
      # self.neumes_baselines[-1].set_upper_bound(self.height)
      # self.neumes_baselines[-1].lyrics.set_lower_bound(self.height)

    def plot_full_baselines(self):
      import matplotlib.pyplot as plt
      import matplotlib.patches as patches

      # Create figure and axes
      fig, ax = plt.subplots(figsize=(self.height / 10, self.width / 10))

      # Display the image
      ax.imshow(self.image)

      self.compute_full_baselines()
      for index, nb in enumerate(self.neumes_baselines):
        rect = patches.Rectangle((0, nb.y), self.master.shape[0], 2.5, linewidth=2.5, edgecolor='purple', facecolor='none', label=f'{index}')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        ax.annotate(f'neumes: {index}', (cx, cy), color='green', weight='bold', fontsize=16, ha='center', va='center')
      
      for index, nb in enumerate(self.neumes_baselines):
        rect = patches.Rectangle((10, nb.lyrics.y), self.master.shape[0] - 10, 2.5, linewidth=2.5, edgecolor='orange', facecolor='none', label=f'{index}')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        ax.annotate(f'lyrics: {index}', (cx, cy), color='green', weight='bold', fontsize=16, ha='center', va='center')
      plt.show()

    def map_neumes(self):
      # Compute the baselines.
      self.compute_full_baselines()

      # And fetch the neumes.
      for nb in self.neumes_baselines:
        nb.fetch_touching_neumes()
      for nb in self.neumes_baselines:
        nb.fetch_final_neumes()

    def plot_mapped_neumes(self):
      import matplotlib.pyplot as plt
      import matplotlib.patches as patches

      # Create figure and axes
      fig, ax = plt.subplots(figsize=(self.height / 10, self.width / 10))

      # Display the image
      ax.imshow(self.image)

      self.map_neumes()
      for index, nb in enumerate(self.neumes_baselines):
        rect = patches.Rectangle((0, nb.y), self.master.shape[0], 2.5, linewidth=2.5, edgecolor='purple', facecolor='none', label=f'{index}')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        # ax.annotate(f'neumes: {index}', (cx, cy), color='green', weight='bold', fontsize=16, ha='center', va='center')
        for ptr in nb.neumes:
          cc = self.ccs[ptr]
          nr = patches.Rectangle((cc.x, cc.y), cc.w, cc.h, linewidth=2, edgecolor='blue', facecolor='none')
          ax.add_patch(nr)
      plt.show()

      # for index, nb in enumerate(self.neumes_baselines):
      #   rect = patches.Rectangle((10, nb.lyrics.y), self.master.shape[0] - 10, 2.5, linewidth=2.5, edgecolor='orange', facecolor='none', label=f'{index}')
      #   ax.add_patch(rect)
      #   rx, ry = rect.get_xy()
      #   cx = rx + rect.get_width() / 2.0
      #   cy = ry + rect.get_height() / 2.0
      #   ax.annotate(f'lyrics: {index}', (cx, cy), color='green', weight='bold', fontsize=16, ha='center', va='center')
      # plt.show()


    # Compute zero ranges.
    def compute_zero_ranges(self):
      # self.thresh[self.thresh == 255] = 1
      hs = np.sum(self.thresh, axis=1) / 255
      # num_occ_oligon = int(np.ceil(self.master.shape[0] / self.master.oligon_width))
      hs[hs < 1] = 0

      ranges = []
      def add_range(pos, t):
        if t == +1:
          ranges.append((pos, -1))
        else:
          assert len(ranges) is not None
          ranges[-1] = (ranges[-1][0], pos)

      inside = 0
      for i in range(len(hs)):
        if hs[i] > 0:
          if inside:
            add_range(i, -1)
            inside = 0
          continue
        if not inside:
          add_range(i, +1)
          inside = 1
      if not hs[-1]:
        add_range(len(hs), -1)
      return ranges   

    def plot_horizontal_projection(self, ratio=kRatio):
      import matplotlib.pyplot as plt
      f = plt.figure()
      f.set_figwidth(50)
      f.set_figheight(10)

      hs = self.compute_horizontal_projection(ratio=ratio)
      self.compute_neumes_baselines()
      for nb in self.neumes_baselines:
        plt.plot([nb.y], [hs[nb.y]], marker='o', markersize=15, color="red")
      plt.plot(hs, color='black')

    def plot_ranges(self):
      import matplotlib.pyplot as plt
      import matplotlib.patches as patches

      # Create figure and axes
      fig, ax = plt.subplots(figsize=(self.height / 10, self.width / 10))

      # Display the image
      ax.imshow(self.image)

      rs = self.compute_zero_ranges()
      for index, (b, e) in enumerate(rs):
        rect = patches.Rectangle((0, b), self.master.shape[0], e - b - 1, linewidth=2, edgecolor='purple', facecolor='purple', label=f'{index}')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        ax.annotate(f'{index}', (cx, cy), color='green', weight='bold', fontsize=16, ha='center', va='center')
      plt.show()

    # Plot the page with all its connected components (could be filtered).
    def plot_ccs(self, ratio=kRatio):
      import matplotlib.pyplot as plt
      import matplotlib.patches as patches

      # Create figure and axes
      fig, ax = plt.subplots(figsize=(self.height / 10, self.width / 10))

      # Display the image
      ax.imshow(self.image)

      # Create a Rectangle patch
      max_x, max_y = 0, 0
      for index, cc in enumerate(self.ccs):
        if cc.w / cc.h < ratio:
          continue
        max_x = max(max_x, cc.x + cc.w)
        max_y = max(max_y, cc.y + cc.h)
        rect = patches.Rectangle((cc.x, cc.y), cc.w, cc.h, linewidth=1, edgecolor='r', facecolor='none', label=f'{index}')
        ax.add_patch(rect)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        ax.annotate(f'{index}', (cx, cy), color='green', weight='bold', fontsize=16, ha='center', va='center')
      plt.show()

  # Sheet constructor.
  def __init__(self, file_path):
    images = to_images(file_path)
    self.pages = []
    for image in images:
      self.pages.append(self.Page(self, image))

    # Compute the shape.
    self.shape = (max(p.width for p in self.pages), max(p.height for p in self.pages))

    # Compute oligon parameters.
    self.oligon_width = self.compute_oligon_width()
    self.oligon_height = self.compute_oligon_height()

  def compute_oligon_width(self, ratio=kRatio):
    ws = [0] * (self.shape[0] + 1)
    for p in self.pages:
      for cc in p.ccs:
        if cc.w / cc.h < ratio:
          continue
        ws[cc.w] += 1
    total = sum(ws)
    curr = 0
    for i in range(1, self.shape[0] + 1):
      curr += ws[i]
      if curr >= total / 2:
        return i + 1
    return None

  def compute_oligon_height(self, ratio=kRatio):
    hs = [0] * (self.shape[1] + 1)
    
    def collect_vertical_runlengths(p, cc):
      for x_ in range(cc.x, cc.x + cc.w):
        sum = 0
        for y_ in range(cc.y, cc.y + cc.h):
          bit = int(p.thresh[y_][x_] == 255)
          if bit == 1:
            sum += 1
          else:
            hs[sum] += 1
            sum = 0
        hs[sum] += 1

    for p in self.pages:
      for cc in p.ccs:
        if cc.w / cc.h < ratio:
          continue
        collect_vertical_runlengths(p, cc)
        
    max_freq = 0
    oligon_height = None
    for i in range(1, self.shape[1] + 1):
      if hs[i] > max_freq:
        max_freq = hs[i]
        oligon_height = i
    assert oligon_height is not None
    return oligon_height

  def __getitem__(self, key):
      return self.pages[key]