from turtle import right
from pkg_resources import yield_lines
import cv2
import numpy as np
from src.util import last_arg, to_images

# Constants.
kRatio = 3

class Sheet:
  # Page class.
  class Page:
    # Baseline:
    class Baseline:
      class Lyrics:
        def __init__(self, master):
          self.master = master

        def set_coordinate(self, y):
          self.y = y

        # TODO: avoid duplicate code -> use inheritance!
        def set_lower_bound(self, lb):
          self.lb = lb

      def __init__(self, master, y):
        self.master = master
        self.y = y
        self.lyrics = self.Lyrics(self)

      def __str__(self):
        return f'y = {self.y}'

      def set_lower_bound(self, lb):
        self.lb = lb

      def set_upper_bound(self, ub):
        self.ub = ub

    # Constructor.
    def __init__(self, master, image):
      self.master = master
      self.image = image
      gray = cv2.cvtColor(np.array(image.convert('RGB'))[:, :, ::-1].copy(), cv2.COLOR_BGR2GRAY)
      self.height, self.width = gray.shape

      self.thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      output = cv2.connectedComponentsWithStats(self.thresh, 8, cv2.CV_32S)
      (_, _, self.ccs, self.centroids) = output

    # Compute the horizontal projection of connected components with `w / h` < `ratio`. 
    def compute_horizontal_projection(self, ratio=kRatio):
      hs = np.zeros(self.master.shape[1] + 1)

      def collect_horizontal_runlengths(x, y, w, h):
        for y_ in range(y, y + h):
          hs[y_] += np.count_nonzero(self.thresh[y_][x : x + w] == 255)

      for (x, y, w, h, _) in self.ccs:
        if w / h < ratio:
          continue
        collect_horizontal_runlengths(x, y, w, h)
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
      for nb in new_peaks:
        self.neumes_baselines.append(self.Baseline(self, nb))
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

      print([str(nb) for nb in self.neumes_baselines])

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

        print(f'b1.y={b1.y}, b2.y={b2.y}, mid={mid}, max_peaks={max_peaks}, values={rhp[max_peaks]}')
        ind = np.argpartition(rhp[max_peaks], -2)[-2:]
        print(f'ind={ind}')

        # TODO: what if `len(ind) == 1`?
        assert len(ind) == 2
        max1, max2 = max_peaks[ind]

        print(f'max1={max1}, max2={max2}')
        b1.lyrics.set_coordinate(max1 + (max2 - max1) / 2)
        
      # TODO: could go wrong, when the end of sentence is really short!
      # TODO: and we also have page numbers.
      self.neumes_baselines.append(self.Baseline(self, self.height))
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
      for index, (x, y, w, h, a) in enumerate(self.ccs):
        if w / h < ratio:
          continue
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none', label=f'{index}')
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
      for (_, _, w, h, _) in p.ccs:
        if w / h < ratio:
          continue
        ws[w] += 1
    total = sum(ws)
    curr = 0
    for i in range(1, self.shape[0] + 1):
      curr += ws[i]
      if curr >= total / 2:
        return i + 1
    return None

  def compute_oligon_height(self, ratio=kRatio):
    hs = [0] * (self.shape[1] + 1)
    
    def collect_vertical_runlengths(p, x, y, w, h):
      for x_ in range(x, x + w):
        sum = 0
        for y_ in range(y, y + h):
          bit = int(p.thresh[y_][x_] == 255)
          if bit == 1:
            sum += 1
          else:
            hs[sum] += 1
            sum = 0
        hs[sum] += 1

    for p in self.pages:
      for (x, y, w, h, _) in p.ccs:
        if w / h < ratio:
          continue
        collect_vertical_runlengths(p, x, y, w, h)
        
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