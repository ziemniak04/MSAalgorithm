import argparse
import numpy as np
from typing import List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Sequence:
    """Class to represent a biological sequence."""

    def __init__(self, identifier: str, sequence: str):
        self.identifier = identifier
        self.sequence = sequence.upper()

    def __str__(self) -> str:
        return f">{self.identifier}\n{self.sequence}"

    def __len__(self) -> int:
        return len(self.sequence)


class PairwiseAlignment:
    """Class to perform pairwise sequence alignment using Needleman-Wunsch algorithm."""

    def __init__(self, match_score: int = 1, mismatch_score: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty

    def score(self, a: str, b: str) -> int:
        """Calculate score for two characters."""
        if a == b:
            return self.match_score
        elif a == '-' or b == '-':
            return self.gap_penalty
        else:
            return self.mismatch_score

    def align(self, seq1: str, seq2: str) -> Tuple[str, str, int]:
        """Perform global alignment using Needleman-Wunsch algorithm."""
        # Initialize scoring matrix
        n, m = len(seq1), len(seq2)
        score_matrix = np.zeros((n + 1, m + 1), dtype=int)

        # Initialize traceback matrix: 0 = diagonal, 1 = up, 2 = left
        traceback = np.zeros((n + 1, m + 1), dtype=int)

        # Initialize first row and column with gap penalties
        for i in range(1, n + 1):
            score_matrix[i, 0] = i * self.gap_penalty
            traceback[i, 0] = 1  # Up
        for j in range(1, m + 1):
            score_matrix[0, j] = j * self.gap_penalty
            traceback[0, j] = 2  # Left

        # Fill the matrices
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = score_matrix[i - 1, j - 1] + self.score(seq1[i - 1], seq2[j - 1])
                delete = score_matrix[i - 1, j] + self.gap_penalty
                insert = score_matrix[i, j - 1] + self.gap_penalty

                # Determine the best move
                score_matrix[i, j] = max(match, delete, insert)

                # Set traceback pointer
                if score_matrix[i, j] == match:
                    traceback[i, j] = 0  # Diagonal
                elif score_matrix[i, j] == delete:
                    traceback[i, j] = 1  # Up
                else:
                    traceback[i, j] = 2  # Left

        # Traceback to find alignment
        align1 = []
        align2 = []
        i, j = n, m

        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i, j] == 0:  # Diagonal
                align1.append(seq1[i - 1])
                align2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and traceback[i, j] == 1:  # Up
                align1.append(seq1[i - 1])
                align2.append('-')
                i -= 1
            else:  # Left
                align1.append('-')
                align2.append(seq2[j - 1])
                j -= 1

        # Reverse the alignments
        align1 = ''.join(reversed(align1))
        align2 = ''.join(reversed(align2))

        # Return the aligned sequences and the alignment score
        return align1, align2, score_matrix[n, m]

    def calculate_stats(self, aligned_seq1: str, aligned_seq2: str) -> Dict:
        """Calculate statistics for the alignment."""
        matches = 0
        mismatches = 0
        gaps = 0

        for i in range(len(aligned_seq1)):
            if aligned_seq1[i] == '-' or aligned_seq2[i] == '-':
                gaps += 1
            elif aligned_seq1[i] == aligned_seq2[i]:
                matches += 1
            else:
                mismatches += 1

        # Calculate identity percentage
        identity = matches / len(aligned_seq1) * 100 if len(aligned_seq1) > 0 else 0

        return {
            'matches': matches,
            'mismatches': mismatches,
            'gaps': gaps,
            'identity': identity
        }


class MultipleSequenceAlignment:
    """Class to perform multiple sequence alignment using the center star method."""

    def __init__(self, match_score: int = 1, mismatch_score: int = -1, gap_penalty: int = -2):
        self.pairwise_aligner = PairwiseAlignment(match_score, mismatch_score, gap_penalty)
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty

    def find_center_sequence(self, sequences: List[Sequence]) -> int:
        """Find the center sequence (the one with highest sum of pairwise scores)."""
        n = len(sequences)
        sum_scores = [0] * n

        for i in range(n):
            for j in range(n):
                if i != j:
                    _, _, score = self.pairwise_aligner.align(sequences[i].sequence, sequences[j].sequence)
                    sum_scores[i] += score

        # Return the index of the sequence with the highest sum of scores
        return sum_scores.index(max(sum_scores))

    def align_to_center(self, center_seq: str, other_seq: str) -> Tuple[str, str]:
        """Align a sequence to the center sequence."""
        return self.pairwise_aligner.align(center_seq, other_seq)[0:2]

    def merge_alignments(self, center_alignments: List[Tuple[str, str]]) -> List[str]:
        """Merge all pairwise alignments with the center sequence into a multiple alignment."""
        # Extract center sequence from the first alignment (it's consistent across all alignments)
        center_seq = center_alignments[0][0]

        # Initialize the multiple alignment with the center sequence
        msa = [center_seq]

        # Process each alignment with the center
        for _, aligned_other in center_alignments:
            # Create a new sequence that will be added to the MSA
            new_seq = list('-' * len(center_seq))

            # Map the aligned_other to the center sequence positions
            center_pos = 0
            for i, char in enumerate(center_seq):
                if char != '-':  # Not a gap in center
                    if center_pos < len(aligned_other) and aligned_other[center_pos] != '-':
                        new_seq[i] = aligned_other[center_pos]
                    center_pos += 1

            msa.append(''.join(new_seq))

        return msa

    def align(self, sequences: List[Sequence]) -> Tuple[List[str], Dict]:
        """Perform multiple sequence alignment using center star method."""
        if len(sequences) <= 1:
            return [seq.sequence for seq in sequences], {}

        # Find the center sequence
        center_idx = self.find_center_sequence(sequences)
        center_sequence = sequences[center_idx]

        # Align each sequence to the center
        center_alignments = []
        for i, seq in enumerate(sequences):
            if i != center_idx:
                aligned_center, aligned_other = self.align_to_center(center_sequence.sequence, seq.sequence)
                center_alignments.append((aligned_center, aligned_other))

        # Insert the self-alignment of the center sequence for consistency
        center_alignments.insert(center_idx, (center_sequence.sequence, center_sequence.sequence))

        # Merge all alignments
        aligned_sequences = self.merge_alignments(center_alignments)

        # Calculate statistics
        stats = self.calculate_msa_stats(aligned_sequences)

        return aligned_sequences, stats

    def calculate_msa_stats(self, aligned_sequences: List[str]) -> Dict:
        """Calculate statistics for the multiple sequence alignment."""
        if not aligned_sequences:
            return {}

        n_seqs = len(aligned_sequences)
        alignment_length = len(aligned_sequences[0])

        total_pairwise_matches = 0
        total_pairwise_mismatches = 0
        total_pairwise_gaps = 0
        total_comparisons = 0

        # Calculate pairwise statistics
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                stats = self.pairwise_aligner.calculate_stats(
                    aligned_sequences[i], aligned_sequences[j]
                )
                total_pairwise_matches += stats['matches']
                total_pairwise_mismatches += stats['mismatches']
                total_pairwise_gaps += stats['gaps']
                total_comparisons += 1

        # Calculate average identity across all pairs
        avg_identity = 0
        if total_comparisons > 0:
            total_positions = alignment_length * total_comparisons
            avg_identity = total_pairwise_matches / total_positions * 100 if total_positions > 0 else 0

        return {
            'avg_identity': avg_identity,
            'total_matches': total_pairwise_matches,
            'total_mismatches': total_pairwise_mismatches,
            'total_gaps': total_pairwise_gaps,
            'alignment_length': alignment_length
        }


class FASTAParser:
    """Class to parse FASTA format files."""

    @staticmethod
    def read_fasta(file_path: str) -> List[Sequence]:
        """Read sequences from a FASTA file."""
        sequences = []
        current_id = ""
        current_seq = []

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    # Save the previous sequence if there is one
                    if current_id and current_seq:
                        sequences.append(Sequence(current_id, ''.join(current_seq)))

                    # Start a new sequence
                    current_id = line[1:]  # Remove the '>' character
                    current_seq = []
                else:
                    # Add to the current sequence
                    current_seq.append(line)

        # Add the last sequence
        if current_id and current_seq:
            sequences.append(Sequence(current_id, ''.join(current_seq)))

        return sequences

    @staticmethod
    def write_fasta(sequences: List[Sequence], file_path: str) -> None:
        """Write sequences to a FASTA file."""
        with open(file_path, 'w') as file:
            for seq in sequences:
                file.write(f">{seq.identifier}\n")
                file.write(f"{seq.sequence}\n")


class MSAOutputWriter:
    """Class to write MSA output to a file."""

    @staticmethod
    def write_alignment(aligned_sequences: List[str], original_sequences: List[Sequence],
                        stats: Dict, params: Dict, file_path: str) -> None:
        """Write alignment and statistics to a file."""
        with open(file_path, 'w') as file:
            # Write parameters
            file.write("MULTIPLE SEQUENCE ALIGNMENT - CENTER STAR METHOD\n")
            file.write("=" * 50 + "\n\n")
            file.write("Parameters:\n")
            file.write(f"Match score: {params['match_score']}\n")
            file.write(f"Mismatch score: {params['mismatch_score']}\n")
            file.write(f"Gap penalty: {params['gap_penalty']}\n\n")

            # Write alignment
            file.write("Alignment:\n")
            for i, (seq, orig_seq) in enumerate(zip(aligned_sequences, original_sequences)):
                file.write(f">{orig_seq.identifier}\n")
                file.write(f"{seq}\n")
            file.write("\n")

            # Write statistics
            file.write("Statistics:\n")
            file.write(f"Average identity: {stats['avg_identity']:.2f}%\n")
            file.write(f"Total matches: {stats['total_matches']}\n")
            file.write(f"Total mismatches: {stats['total_mismatches']}\n")
            file.write(f"Total gaps: {stats['total_gaps']}\n")
            file.write(f"Alignment length: {stats['alignment_length']}\n")


class ConsoleInterface:
    """Console interface for the MSA program."""

    def __init__(self):
        self.sequences = []
        self.match_score = 1
        self.mismatch_score = -1
        self.gap_penalty = -2

    def get_param_input(self) -> None:
        """Get scoring parameters from user input."""
        try:
            self.match_score = int(input("Enter match score (default 1): ") or "1")
            self.mismatch_score = int(input("Enter mismatch score (default -1): ") or "-1")
            self.gap_penalty = int(input("Enter gap penalty (default -2): ") or "-2")
        except ValueError:
            print("Invalid input. Using default values.")

    def get_sequence_input(self) -> None:
        """Get sequences from user input."""
        num_sequences = int(input("Enter number of sequences: "))
        for i in range(num_sequences):
            identifier = input(f"Enter identifier for sequence {i + 1}: ")
            sequence = input(f"Enter sequence {i + 1}: ").strip()
            self.sequences.append(Sequence(identifier, sequence))

    def load_from_file(self) -> None:
        """Load sequences from a FASTA file."""
        file_path = input("Enter FASTA file path: ")
        try:
            self.sequences = FASTAParser.read_fasta(file_path)
            print(f"Loaded {len(self.sequences)} sequences from file.")
        except Exception as e:
            print(f"Error loading file: {e}")

    def run(self) -> None:
        """Run the MSA program."""
        print("MULTIPLE SEQUENCE ALIGNMENT - CENTER STAR METHOD")
        print("=" * 50)

        # Get input method
        print("\nInput method:")
        print("1. Manual input")
        print("2. Load from FASTA file")
        choice = input("Choose option (1 or 2): ")

        if choice == "1":
            self.get_sequence_input()
        elif choice == "2":
            self.load_from_file()
        else:
            print("Invalid choice. Exiting.")
            return

        if not self.sequences:
            print("No sequences loaded. Exiting.")
            return

        # Get scoring parameters
        self.get_param_input()

        # Perform MSA
        msa = MultipleSequenceAlignment(self.match_score, self.mismatch_score, self.gap_penalty)
        aligned_sequences, stats = msa.align(self.sequences)

        # Display results
        print("\nAlignment:")
        for i, (seq, aligned) in enumerate(zip(self.sequences, aligned_sequences)):
            print(f">{seq.identifier}")
            print(aligned)

        print("\nStatistics:")
        print(f"Average identity: {stats['avg_identity']:.2f}%")
        print(f"Total matches: {stats['total_matches']}")
        print(f"Total mismatches: {stats['total_mismatches']}")
        print(f"Total gaps: {stats['total_gaps']}")
        print(f"Alignment length: {stats['alignment_length']}")

        # Save to file
        save_option = input("\nDo you want to save the alignment to a file? (y/n): ")
        if save_option.lower() == 'y':
            output_file = input("Enter output file path: ")
            params = {
                'match_score': self.match_score,
                'mismatch_score': self.mismatch_score,
                'gap_penalty': self.gap_penalty
            }
            MSAOutputWriter.write_alignment(aligned_sequences, self.sequences, stats, params, output_file)
            print(f"Alignment saved to {output_file}")


class GUI(tk.Tk):
    """GUI for the MSA program."""

    def __init__(self):
        super().__init__()

        self.title("Multiple Sequence Alignment - Center Star Method")
        self.geometry("900x700")
        self.minsize(800, 600)

        self.sequences = []
        self.aligned_sequences = []
        self.stats = {}

        self.create_widgets()

    def create_widgets(self):
        """Create all widgets for the GUI."""
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        # Buttons for input
        ttk.Button(input_frame, text="Load FASTA File", command=self.load_fasta).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Enter Sequences Manually", command=self.show_manual_input).pack(side=tk.LEFT,
                                                                                                      padx=5)

        # Parameters frame
        param_frame = ttk.LabelFrame(main_frame, text="Scoring Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=5)

        # Match score
        ttk.Label(param_frame, text="Match Score:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.match_var = tk.StringVar(value="1")
        ttk.Entry(param_frame, textvariable=self.match_var, width=5).grid(row=0, column=1, padx=5, pady=5)

        # Mismatch score
        ttk.Label(param_frame, text="Mismatch Score:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.mismatch_var = tk.StringVar(value="-1")
        ttk.Entry(param_frame, textvariable=self.mismatch_var, width=5).grid(row=0, column=3, padx=5, pady=5)

        # Gap penalty
        ttk.Label(param_frame, text="Gap Penalty:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.gap_var = tk.StringVar(value="-2")
        ttk.Entry(param_frame, textvariable=self.gap_var, width=5).grid(row=0, column=5, padx=5, pady=5)

        # Sequence display frame
        seq_frame = ttk.LabelFrame(main_frame, text="Sequences", padding="10")
        seq_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Sequences text widget
        self.seq_text = tk.Text(seq_frame, height=10, width=80, wrap=tk.NONE)
        self.seq_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for sequences
        seq_scrolly = ttk.Scrollbar(seq_frame, orient=tk.VERTICAL, command=self.seq_text.yview)
        seq_scrolly.pack(side=tk.RIGHT, fill=tk.Y)
        self.seq_text.config(yscrollcommand=seq_scrolly.set)

        seq_scrollx = ttk.Scrollbar(seq_frame, orient=tk.HORIZONTAL, command=self.seq_text.xview)
        seq_scrollx.pack(side=tk.BOTTOM, fill=tk.X)
        self.seq_text.config(xscrollcommand=seq_scrollx.set)

        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=5)

        ttk.Button(action_frame, text="Run Alignment", command=self.run_alignment).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Alignment Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Notebook for results display
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab for text display
        text_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(text_tab, text="Text View")

        # Results text widget
        self.results_text = tk.Text(text_tab, height=15, width=80, wrap=tk.NONE)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for results
        results_scrolly = ttk.Scrollbar(text_tab, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrolly.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrolly.set)

        results_scrollx = ttk.Scrollbar(text_tab, orient=tk.HORIZONTAL, command=self.results_text.xview)
        results_scrollx.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_text.config(xscrollcommand=results_scrollx.set)

        # Tab for visual display
        visual_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(visual_tab, text="Visual View")

        # Frame for matplotlib figure
        self.figure_frame = ttk.Frame(visual_tab)
        self.figure_frame.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var.set("Ready")

    def load_fasta(self):
        """Load sequences from a FASTA file."""
        file_path = filedialog.askopenfilename(
            title="Select FASTA File",
            filetypes=[("FASTA files", "*.fasta *.fa *.fna *.ffn *.faa *.frn"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.sequences = FASTAParser.read_fasta(file_path)
            self.display_sequences()
            self.status_var.set(f"Loaded {len(self.sequences)} sequences from file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load FASTA file: {e}")

    def show_manual_input(self):
        """Show dialog for manual sequence input."""
        dialog = tk.Toplevel(self)
        dialog.title("Enter Sequences")
        dialog.geometry("600x400")
        dialog.transient(self)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Enter sequences in FASTA format:").pack(anchor=tk.W)

        text = tk.Text(frame, height=15, width=60)
        text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Sample placeholder
        placeholder = """>Sequence1
ACGTACGTACGT
>Sequence2
ACGTACCTACGT
>Sequence3
ACGTACGTAGGT"""
        text.insert(tk.END, placeholder)

        def submit():
            fasta_text = text.get("1.0", tk.END)

            # Parse sequences from text
            self.sequences = []
            current_id = ""
            current_seq = []

            for line in fasta_text.splitlines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    # Save the previous sequence if there is one
                    if current_id and current_seq:
                        self.sequences.append(Sequence(current_id, ''.join(current_seq)))

                    # Start a new sequence
                    current_id = line[1:]  # Remove the '>' character
                    current_seq = []
                else:
                    # Add to the current sequence
                    current_seq.append(line)

            # Add the last sequence
            if current_id and current_seq:
                self.sequences.append(Sequence(current_id, ''.join(current_seq)))

            self.display_sequences()
            self.status_var.set(f"Added {len(self.sequences)} sequences.")
            dialog.destroy()

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Submit", command=submit).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def display_sequences(self):
        """Display loaded sequences in the text widget."""
        self.seq_text.delete("1.0", tk.END)

        for seq in self.sequences:
            self.seq_text.insert(tk.END, f">{seq.identifier}\n{seq.sequence}\n\n")

    def run_alignment(self):
        """Run the MSA algorithm."""
        if not self.sequences:
            messagebox.showwarning("No Sequences", "Please load or enter sequences first.")
            return

        try:
            match_score = int(self.match_var.get())
            mismatch_score = int(self.mismatch_var.get())
            gap_penalty = int(self.gap_var.get())
        except ValueError:
            messagebox.showerror("Invalid Parameters", "Please enter valid scoring parameters.")
            return

        # Perform MSA
        self.status_var.set("Running alignment...")
        self.update_idletasks()

        msa = MultipleSequenceAlignment(match_score, mismatch_score, gap_penalty)
        self.aligned_sequences, self.stats = msa.align(self.sequences)

        # Display results
        self.display_results()
        self.create_visualization()

        self.status_var.set("Alignment complete")

    def display_results(self):
        """Display alignment results in the text widget."""
        self.results_text.delete("1.0", tk.END)

        # Display parameters
        self.results_text.insert(tk.END, "MULTIPLE SEQUENCE ALIGNMENT - CENTER STAR METHOD\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")

        self.results_text.insert(tk.END, "Parameters:\n")
        self.results_text.insert(tk.END, f"Match score: {self.match_var.get()}\n")
        self.results_text.insert(tk.END, f"Mismatch score: {self.mismatch_var.get()}\n")
        self.results_text.insert(tk.END, f"Gap penalty: {self.gap_var.get()}\n\n")

        # Display alignment
        self.results_text.insert(tk.END, "Alignment:\n")
        for i, (seq, aligned) in enumerate(zip(self.sequences, self.aligned_sequences)):
            self.results_text.insert(tk.END, f">{seq.identifier}\n")
            self.results_text.insert(tk.END, f"{aligned}\n")

        self.results_text.insert(tk.END, "\n")

        # Display statistics
        self.results_text.insert(tk.END, "Statistics:\n")
        self.results_text.insert(tk.END, f"Average identity: {self.stats['avg_identity']:.2f}%\n")
        self.results_text.insert(tk.END, f"Total matches: {self.stats['total_matches']}\n")
        self.results_text.insert(tk.END, f"Total mismatches: {self.stats['total_mismatches']}\n")
        self.results_text.insert(tk.END, f"Total gaps: {self.stats['total_gaps']}\n")
        self.results_text.insert(tk.END, f"Alignment length: {self.stats['alignment_length']}\n")

    def create_visualization(self):
        """Create visual representation of the alignment."""
        # Clear previous figure
        for widget in self.figure_frame.winfo_children():
            widget.destroy()

        if not self.aligned_sequences:
            return

        # Create matplotlib figure
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # Set up the sequence colors and symbols
        seq_colors = {
            'A': '#00CC00',  # Green
            'C': '#0000CC',  # Blue
            'G': '#FFB300',  # Orange
            'T': '#CC0000',  # Red
            'U': '#CC0000',  # Red (same as T)
            '-': '#FFFFFF',  # White (gap)
        }

        # Default color for unknown residues
        default_color = '#AAAAAA'  # Gray

        # Calculate alignment length
        align_len = len(self.aligned_sequences[0])
        num_seqs = len(self.aligned_sequences)

        # Create an array of colors for each position
        color_array = np.zeros((num_seqs, align_len, 3))

        # Fill the color array
        for i, seq in enumerate(self.aligned_sequences):
            for j, char in enumerate(seq):
                # Get color for this residue
                hex_color = seq_colors.get(char, default_color)

                # Convert hex to RGB (0-1 range)
                r = int(hex_color[1:3], 16) / 255.0
                g = int(hex_color[3:5], 16) / 255.0
                b = int(hex_color[5:7], 16) / 255.0

                color_array[i, j] = [r, g, b]

        # Display the alignment as an image
        ax.imshow(color_array, aspect='auto', interpolation='nearest')

        # Set y-ticks to sequence identifiers
        ax.set_yticks(range(num_seqs))
        ax.set_yticklabels([seq.identifier for seq in self.sequences])

        # Set x-ticks every 10 positions
        x_ticks = range(0, align_len, 10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, align_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_seqs, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)

        # Add title and labels
        ax.set_title('Multiple Sequence Alignment')
        ax.set_xlabel('Position')
        ax.set_ylabel('Sequence')

        # Create a legend for nucleotide/amino acid colors
        legend_elements = []
        for char, color in seq_colors.items():
            if char == '-':
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'Gap ({char})'))
            else:
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=char))

        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=len(seq_colors))

        # Adjust layout
        fig.tight_layout()

        # Embed the figure in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self):
        """Save alignment results to a file."""
        if not self.aligned_sequences:
            messagebox.showwarning("No Results", "Please run alignment first.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Alignment Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            params = {
                'match_score': int(self.match_var.get()),
                'mismatch_score': int(self.mismatch_var.get()),
                'gap_penalty': int(self.gap_var.get())
            }
            MSAOutputWriter.write_alignment(self.aligned_sequences, self.sequences,
                                            self.stats, params, file_path)
            self.status_var.set(f"Results saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def clear_all(self):
        """Clear all data and reset the interface."""
        self.sequences = []
        self.aligned_sequences = []
        self.stats = {}

        self.seq_text.delete("1.0", tk.END)
        self.results_text.delete("1.0", tk.END)

        for widget in self.figure_frame.winfo_children():
            widget.destroy()

        self.status_var.set("Ready")


def main():
    """Main function to run the MSA program."""
    parser = argparse.ArgumentParser(description="Multiple Sequence Alignment - Center Star Method")
    parser.add_argument("--gui", action="store_true", help="Run with graphical user interface")
    parser.add_argument("--fasta", type=str, help="Input FASTA file path")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--match", type=int, default=1, help="Match score")
    parser.add_argument("--mismatch", type=int, default=-1, help="Mismatch score")
    parser.add_argument("--gap", type=int, default=-2, help="Gap penalty")

    args = parser.parse_args()

    if args.gui:
        app = GUI()
        app.mainloop()
    else:
        if args.fasta:
            # Run in command line mode with file input
            try:
                sequences = FASTAParser.read_fasta(args.fasta)
                msa = MultipleSequenceAlignment(args.match, args.mismatch, args.gap)
                aligned_sequences, stats = msa.align(sequences)

                # Display or save results
                if args.output:
                    params = {
                        'match_score': args.match,
                        'mismatch_score': args.mismatch,
                        'gap_penalty': args.gap
                    }
                    MSAOutputWriter.write_alignment(aligned_sequences, sequences, stats, params, args.output)
                    print(f"Results saved to {args.output}")
                else:
                    # Print to console
                    print("\nAlignment:")
                    for i, (seq, aligned) in enumerate(zip(sequences, aligned_sequences)):
                        print(f">{seq.identifier}")
                        print(aligned)

                    print("\nStatistics:")
                    print(f"Average identity: {stats['avg_identity']:.2f}%")
                    print(f"Total matches: {stats['total_matches']}")
                    print(f"Total mismatches: {stats['total_mismatches']}")
                    print(f"Total gaps: {stats['total_gaps']}")
                    print(f"Alignment length: {stats['alignment_length']}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            # Run interactive console interface
            console = ConsoleInterface()
            console.run()


if __name__ == "__main__":
    main()