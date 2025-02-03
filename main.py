import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import threading
import time
from typing import List, Tuple, Dict
import random

# --------------------------
# Part 1: Signal Processing with Error Handling and Magnetism
# --------------------------
class FarOutTV:
    def __init__(self, 
                 array_length: int = 4096,  
                 frequency_range: Tuple[float, float] = (1e3, 5e11),  
                 sampling_rate: float = 1e12):  
        """
        Initialize the FarOutTV Deep Space Signal Scanner.
        
        Args:
            array_length: Length of the antenna array.
            frequency_range: Tuple of min and max frequencies to scan (Hz).
            sampling_rate: Sampling rate of the signal processing system.
        """
        self.array_length = array_length
        self.frequency_range = frequency_range
        self.sampling_rate = sampling_rate
        
        # Inverse phase array configuration
        self.phase_matrix = self._generate_inverse_phase_matrix()
        
        # Define wavebands (frequencies exceeding Nyquist will be skipped)
        self.wavebands = {
            'Radio': (1e3, 1e9),
            'Microwave': (1e9, 3e12),
            'Infrared': (3e12, 4.3e14),
            'Visible': (4.3e14, 7.5e14),
            'Ultraviolet': (7.5e14, 3e16),
            'X-ray': (3e16, 3e19),
            'Gamma-ray': (3e19, 1e25)
        }
        
        # Initialize voxel grid parameters (for visualization purposes)
        self.voxel_grid_size = (100, 100, 100)  # (x, y, z)
        self.voxel_density = np.zeros(self.voxel_grid_size)
    
    def _generate_inverse_phase_matrix(self) -> np.ndarray:
        """
        Generate an inverse phase matrix for signal correlation.
        
        Returns:
            A numpy array representing the phase correlation matrix.
        """
        phase_matrix = np.ones((self.array_length, self.array_length), dtype=complex)
        # Apply helical phase shifts across the array
        for i in range(self.array_length):
            phase_shift = np.exp(1j * 2 * np.pi * i / self.array_length)
            phase_matrix[i, :] *= phase_shift
        # Alternate phase inversion for enhancement
        for i in range(self.array_length):
            if i % 2 == 0:
                phase_matrix[i, :] *= -1
        return phase_matrix
    
    def apply_inverse_phase_scanning(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Apply inverse phase scanning to the raw signal.
        
        Args:
            raw_signal: Input signal array.
        
        Returns:
            Processed signal after inverse phase correlation (magnitude spectrum).
        """
        correlated_signal = np.dot(self.phase_matrix, raw_signal)
        fft_signal = np.fft.fft(correlated_signal)
        return np.abs(fft_signal)
    
    def apply_phase_offset_waveform(self, signal_in: np.ndarray, offset: float) -> np.ndarray:
        """
        Embed a phase offset into the signal for orbital tracking.
        
        Args:
            signal_in: Input signal array.
            offset: Phase offset value.
        
        Returns:
            The phase-shifted signal.
        """
        phase_offset = np.exp(1j * 2 * np.pi * offset)
        return signal_in * phase_offset
    
    def apply_magnetic_transmission(self, signal_in: np.ndarray, magnetic_strength: float = 1.0) -> np.ndarray:
        """
        Apply magnetism as a transmission method.
        This simulates magnetic modulation by multiplying the signal with a
        low-frequency sine wave representing magnetic field variations.
        
        Args:
            signal_in: Input signal array.
            magnetic_strength: Factor controlling the modulation strength.
        
        Returns:
            Magnetically modulated signal.
        """
        t = np.arange(len(signal_in)) / self.sampling_rate
        # Example: 50 Hz modulation (adjustable as needed)
        magnetic_modulator = np.sin(2 * np.pi * 50 * t) * magnetic_strength
        modulated_signal = signal_in * magnetic_modulator
        return modulated_signal
    
    def detect_and_retransmit_signal(self, raw_signal: np.ndarray, ground: bool = True, transmission_method: str = "default") -> np.ndarray:
        """
        Detect and retransmit the signal with minimal transmission.
        
        Args:
            raw_signal: The input signal array.
            ground: If True, the processed signal is transmitted to the ground.
            transmission_method: "default" (or orbital if ground is False) or "magnetism" for magnetic modulation.
        
        Returns:
            The processed and retransmitted signal.
        """
        processed_signal = self.apply_inverse_phase_scanning(raw_signal)
        if transmission_method == "magnetism":
            processed_signal = self.apply_magnetic_transmission(processed_signal)
        elif not ground:
            offset = np.random.random()
            processed_signal = self.apply_phase_offset_waveform(processed_signal, offset)
        if ground:
            processed_signal = -processed_signal
        return processed_signal
    
    def bandpass_filter(self, raw_signal: np.ndarray, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
        """
        Apply a band-pass filter to the raw signal.
        
        Args:
            raw_signal: Input signal.
            lowcut: Low frequency cutoff (Hz).
            highcut: High frequency cutoff (Hz).
            order: Filter order.
        
        Returns:
            The filtered signal.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        high = min(high, 0.99)  # ensure valid cutoff
        if low <= 0 or low >= high:
            raise ValueError(f"Invalid filter parameters: lowcut={lowcut}, highcut={highcut}")
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_signal = signal.lfilter(b, a, raw_signal)
        return filtered_signal
    
    def scan_frequencies(self, raw_signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Scan the raw signal across defined wavebands.
        
        Args:
            raw_signal: Input signal array.
        
        Returns:
            A dictionary mapping waveband names to their processed FFT spectra.
        """
        spectra = {}
        nyquist = 0.5 * self.sampling_rate
        for band, (low, high) in self.wavebands.items():
            if low >= nyquist:
                print(f"Skipping band {band} because lowcut {low} exceeds Nyquist ({nyquist}).")
                continue
            # Adjust highcut to not exceed Nyquist
            high = min(high, nyquist)
            if low >= high:
                print(f"Skipping band {band} because normalized lowcut >= highcut: low={low}, high={high}")
                continue
            try:
                filtered = self.bandpass_filter(raw_signal, low, high)
                processed = self.apply_inverse_phase_scanning(filtered)
                spectra[band] = processed
            except ValueError as e:
                print(f"Skipping band {band} due to error: {e}")
        return spectra
    
    def detect_television_signals(self, fft_signal: np.ndarray, threshold: float = 0.8) -> List[Dict[str, float]]:
        """
        Detect television-like signals in the FFT spectrum.
        
        Args:
            fft_signal: The FFT-processed signal.
            threshold: Sensitivity threshold for peak detection.
        
        Returns:
            A list of detected signal characteristics.
        """
        normalized_signal = fft_signal / np.max(fft_signal)
        peaks, _ = signal.find_peaks(normalized_signal, height=threshold, distance=10)
        detected_signals = []
        for peak in peaks:
            signal_info = {
                'frequency': peak * (self.sampling_rate / self.array_length),
                'intensity': normalized_signal[peak],
                'bandwidth': self._estimate_bandwidth(normalized_signal, peak)
            }
            detected_signals.append(signal_info)
        return detected_signals
    
    def _estimate_bandwidth(self, sig: np.ndarray, peak: int, width_threshold: float = 0.5) -> float:
        """
        Estimate the bandwidth of a detected signal.
        
        Args:
            sig: Processed signal array.
            peak: Index of the signal peak.
            width_threshold: Relative amplitude threshold for bandwidth estimation.
        
        Returns:
            Estimated bandwidth in Hz.
        """
        normalized_peak = sig[peak]
        bandwidth_indices = np.where(sig >= normalized_peak * width_threshold)[0]
        return len(bandwidth_indices) * (self.sampling_rate / self.array_length)
    
    def voxel_scan(self, position: Tuple[int, int, int]) -> None:
        """
        Update the voxel density map based on a scanned position.
        
        Args:
            position: Tuple (x, y, z) representing the voxel coordinates.
        """
        x, y, z = position
        if (0 <= x < self.voxel_grid_size[0] and
            0 <= y < self.voxel_grid_size[1] and
            0 <= z < self.voxel_grid_size[2]):
            self.voxel_density[x, y, z] += 1
    
    def generate_random_position(self) -> Tuple[int, int, int]:
        """
        Generate a random voxel position within the grid.
        
        Returns:
            A tuple (x, y, z) with random voxel indices.
        """
        return tuple(np.random.randint(0, size) for size in self.voxel_grid_size)
    
    def visualize_voxel_density(self) -> None:
        """
        Visualize the 3D voxel density as a 2D projection (summing along z-axis).
        """
        density_2d = np.sum(self.voxel_density, axis=2)
        plt.figure(figsize=(8, 6))
        plt.imshow(density_2d, cmap='hot', interpolation='nearest')
        plt.title('Voxel Density Map')
        plt.xlabel('X Voxel')
        plt.ylabel('Y Voxel')
        plt.colorbar(label='Density')
        plt.show()
    
    def visualize_signal_spectrum(self, fft_signal: np.ndarray):
        """
        Visualize the FFT spectrum of the processed signal.
        
        Args:
            fft_signal: The FFT output of the processed signal.
        """
        plt.figure(figsize=(12, 6))
        frequencies = np.fft.fftfreq(self.array_length, d=1/self.sampling_rate)
        positive_freqs = frequencies[:self.array_length // 2]
        positive_fft = fft_signal[:self.array_length // 2]
        plt.plot(positive_freqs, positive_fft)
        plt.title('Signal Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Signal Intensity')
        plt.xscale('log')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.show()

# --------------------------
# Part 2: Flask Web Server, Database Integration, and Background Scanning
# --------------------------
from flask import Flask, render_template_string, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farouttv.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Models
class Signal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frequency = db.Column(db.Float, nullable=False)
    intensity = db.Column(db.Float, nullable=False)
    bandwidth = db.Column(db.Float, nullable=False)
    region_x = db.Column(db.Integer, nullable=False)
    region_y = db.Column(db.Integer, nullable=False)
    region_z = db.Column(db.Integer, nullable=False)
    video_stream_url = db.Column(db.String(255))

class VoxelDensity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    grid_x = db.Column(db.Integer, nullable=False)
    grid_y = db.Column(db.Integer, nullable=False)
    grid_z = db.Column(db.Integer, nullable=False)
    density = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()

# Instantiate the FarOutTV scanner
scanner = FarOutTV()

def scanning_thread():
    """
    Background thread to simulate continuous signal scanning.
    Runs within an application context so that database operations are permitted.
    """
    while True:
        with app.app_context():
            try:
                # Simulate a composite raw signal (replace with real data as needed)
                t = np.arange(scanner.array_length) / scanner.sampling_rate
                raw_signal = (
                    1e-3 * np.sin(2 * np.pi * 1e5 * t) +        # 100 kHz (Radio)
                    5e-4 * np.sin(2 * np.pi * 5e6 * t) +          # 5 MHz (Microwave)
                    2e-4 * np.sin(2 * np.pi * 1e9 * t) +          # 1 GHz (Microwave)
                    1e-4 * np.sin(2 * np.pi * 1e12 * t)           # 1 THz (Infrared)
                )
                raw_signal += 1e-5 * np.random.randn(scanner.array_length)
                
                spectra = scanner.scan_frequencies(raw_signal)
            except Exception as e:
                print(f"Error during scanning frequencies: {e}")
                time.sleep(1)
                continue
            
            # Process detected signals in the Radio band as an example
            if 'Radio' in spectra:
                radio_spectrum = spectra['Radio']
                detected_signals = scanner.detect_television_signals(radio_spectrum)
                for sig_info in detected_signals:
                    region = scanner.generate_random_position()
                    sig_info['region'] = region
                    sig_info['video_stream_url'] = f"http://example.com/video_{np.random.randint(1000)}.mp4"
                    
                    new_signal = Signal(
                        frequency=sig_info['frequency'],
                        intensity=sig_info['intensity'],
                        bandwidth=sig_info['bandwidth'],
                        region_x=region[0],
                        region_y=region[1],
                        region_z=region[2],
                        video_stream_url=sig_info['video_stream_url']
                    )
                    db.session.add(new_signal)
                    db.session.commit()
                    scanner.voxel_scan(region)
            
            # Demonstrate transmission using a random method (default or magnetism)
            method = random.choice(["default", "magnetism"])
            # Example demodulation (using Hilbert transform for AM)
            demodulated = np.abs(signal.hilbert(raw_signal))
            transmitted = scanner.detect_and_retransmit_signal(raw_signal, ground=True, transmission_method=method)
            # (Optional) You could visualize or further process the transmitted signal.
            
            # Store voxel density data in the database
            density_data = scanner.voxel_density.copy()
            for x in range(scanner.voxel_grid_size[0]):
                for y in range(scanner.voxel_grid_size[1]):
                    for z in range(scanner.voxel_grid_size[2]):
                        density = density_data[x, y, z]
                        if density > 0:
                            voxel = VoxelDensity(
                                grid_x=x,
                                grid_y=y,
                                grid_z=z,
                                density=density
                            )
                            db.session.add(voxel)
            db.session.commit()
            # Reset voxel density after updating
            scanner.voxel_density = np.zeros(scanner.voxel_grid_size)
            time.sleep(5)

# Start scanning in a separate thread
threading.Thread(target=scanning_thread, daemon=True).start()

# HTML Template for the Dashboard (with Three.js visualization)
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>FarOutTV Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; }
        #voxel-map { width: 800px; height: 600px; border: 1px solid #ccc; }
        #video-player { display: none; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>FarOutTV Scanner Dashboard</h1>
    <div id="voxel-map"></div>
    <div id="controls">
        <label for="layer-slider">Layer (Z-axis): <span id="layer-value">50</span></label>
        <input type="range" id="layer-slider" min="0" max="99" value="50">
    </div>
    <div id="video-player">
        <h2>Video Stream</h2>
        <video id="video" width="640" height="480" controls>
            <source id="video-source" src="" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    <!-- Include Three.js from CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 800 / 600, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(800, 600);
        document.getElementById('voxel-map').appendChild(renderer.domElement);
        camera.position.z = 150;
        const voxelSize = 1;
        fetch('/api/get_voxel_density')
            .then(response => response.json())
            .then(data => {
                data.forEach(voxel => {
                    const { grid_x, grid_y, grid_z, density } = voxel;
                    const colorValue = Math.min(density / 10, 1);
                    const color = new THREE.Color(`hsl(${(1 - colorValue) * 240}, 100%, 50%)`);
                    const geometry = new THREE.BoxGeometry(voxelSize, voxelSize, voxelSize);
                    const material = new THREE.MeshBasicMaterial({ color: color });
                    const cube = new THREE.Mesh(geometry, material);
                    cube.position.set(grid_x * voxelSize, grid_y * voxelSize, grid_z * voxelSize);
                    scene.add(cube);
                });
                renderer.render(scene, camera);
            });
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        animate();
        const layerSlider = document.getElementById('layer-slider');
        const layerValue = document.getElementById('layer-value');
        layerSlider.addEventListener('input', () => {
            const selectedLayer = parseInt(layerSlider.value);
            layerValue.textContent = selectedLayer;
            scene.children.forEach(cube => {
                if (Math.round(cube.position.z / voxelSize) === selectedLayer) {
                    cube.visible = true;
                } else {
                    cube.visible = false;
                }
            });
        });
        layerSlider.dispatchEvent(new Event('input'));
        renderer.domElement.addEventListener('click', (event) => {
            const mouse = new THREE.Vector2();
            mouse.x = (event.clientX / renderer.domElement.clientWidth) * 2 - 1;
            mouse.y = - (event.clientY / renderer.domElement.clientHeight) * 2 + 1;
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(scene.children);
            if (intersects.length > 0) {
                const intersect = intersects[0].object;
                const pos = intersect.position;
                const grid_x = Math.round(pos.x / voxelSize);
                const grid_y = Math.round(pos.y / voxelSize);
                const grid_z = Math.round(pos.z / voxelSize);
                fetch(`/api/get_signals_by_region?x=${grid_x}&y=${grid_y}&z=${grid_z}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.signals.length > 0) {
                            const videoUrl = data.signals[0].video_stream_url;
                            if (videoUrl) {
                                document.getElementById('video-source').src = videoUrl;
                                document.getElementById('video').load();
                                document.getElementById('video-player').style.display = 'block';
                            } else {
                                alert('No video stream available for this signal.');
                            }
                        } else {
                            alert('No signals found in this region.');
                        }
                    });
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/api/get_voxel_density')
def get_voxel_density():
    voxel_data = VoxelDensity.query.all()
    voxel_list = [{
        'grid_x': voxel.grid_x,
        'grid_y': voxel.grid_y,
        'grid_z': voxel.grid_z,
        'density': voxel.density
    } for voxel in voxel_data]
    return jsonify(voxel_list)

@app.route('/api/get_signals_by_region')
def get_signals_by_region():
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    z = request.args.get('z', type=int)
    signals = Signal.query.filter(
        and_(
            Signal.region_x == x,
            Signal.region_y == y,
            Signal.region_z == z
        )
    ).all()
    signal_list = [{
        'id': signal.id,
        'frequency': signal.frequency,
        'intensity': signal.intensity,
        'bandwidth': signal.bandwidth,
        'video_stream_url': signal.video_stream_url
    } for signal in signals]
    return jsonify({'signals': signal_list})

if __name__ == "__main__":
    # Disable the reloader to avoid subprocess PermissionError issues.
    app.run(debug=True, use_reloader=False)
