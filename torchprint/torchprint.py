from rich import print
nativeprint = print
from rich.table import Table
from rich.console import Console
import torch
import numpy as np
import lovely_tensors as lt
from typing import Any

def channels(t: torch.Tensor, **kwargs: Any) -> None:
    """
    Display tensor channels using lovely_tensors.

    Args:
        t (torch.Tensor): The tensor to display.
        **kwargs (Any): Additional keyword arguments passed to lt.chans().
    """
    # Display the tensor channels using lovely_tensors
    display(lt.chans(t, **kwargs))


def rgb(t: torch.Tensor, **kwargs: Any) -> None:
    """
    Display tensor as RGB image using lovely_tensors.

    Args:
        t (torch.Tensor): The tensor to display.
        **kwargs (Any): Additional keyword arguments passed to lt.rgb().
    """
    # Display the tensor as an RGB image using lovely_tensors
    display(lt.rgb(t, **kwargs))


def tprint(args, shape=False, dtype=False, device=False, grad_fn=False, **kwargs):

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    output = []
    np.set_printoptions(precision=4, suppress=True)

    def tensor_to_string(tensor):
        return str(tensor.cpu().detach().numpy())

    for arg in args:
        if isinstance(arg, torch.Tensor):

            infos = ""
            if shape:
                infos += f"Shape: {tuple(arg.shape)}"
            if dtype:
                infos += f"Dtype {str(arg.dtype).split('torch.')[1]}"
            if device:
                infos += f"Device: {arg.device}"
            if grad_fn:
                infos += (
                    f"Grad_fn: {arg.grad_fn}" if arg.grad_fn is not None else "NOGRAD"
                )
            if shape or dtype or device or grad_fn:
                infos += "\n"
            infos += tensor_to_string(arg)

            output.append(infos)
        elif (isinstance(arg, list) or isinstance(arg, tuple)) and all(
            isinstance(x, torch.Tensor) for x in arg
        ):
            print(f"{len(arg)} elements:", [x.shape for x in arg])
        else:
            output.append(str(arg))
    nativeprint(sep.join(output), end=end)

def display_table(string1, string2):
    # Create a console object
    console = Console()

    # Create a table with no borders
    table = Table(show_header=False, show_edge=False, show_lines=False, box=None)
    
    # Add columns with respective widthstma
    table.add_column("Column 1", width=80,)
    table.add_column("Column 2", width=20,)

    # Add the row with the two strings
    table.add_row(str(string1), str(string2))
    
    # Return the table as a string
    return table
    # Print the table to the console
    # console.print(table)

def print(*args: Any, **kwargs: Any) -> None:
    """
    Custom print function to handle both tensors and regular objects.

    If the argument is a torch.Tensor, use lovely_tensors to print.
    Otherwise, use the rich print function.

    Args:
        *args (Any): The arguments to print.
        **kwargs (Any): Additional keyword arguments passed to the print function.
    """
    # Check if any argument is a torch.Tensor
    if any(isinstance(arg, torch.Tensor) for arg in args):
        tprint(args, **kwargs)
    else:
        # Use the original print function for non-tensor arguments
        nativeprint(*args, **kwargs)

import torch.nn.functional as F
def pad_tensor_with_random_values(tensor, pad_size):
    # Create a list of padding sizes for each dimension (2 * pad_size for each dimension)
    pad = []
    for _ in range(tensor.dim()):
        pad.extend([pad_size, pad_size])
    
    # Pad the tensor with random values
    padded_tensor = F.pad(tensor, pad, mode='constant', value=0)
    
    # Replace the padding with random values
    for dim in range(tensor.dim()):
        pad_slices = [slice(None)] * tensor.dim()
        pad_slices[dim] = slice(0, pad_size)
        padded_tensor[pad_slices] = torch.arange(0,tensor.size(dim))
        
        pad_slices[dim] = slice(-pad_size, None)
        padded_tensor[pad_slices] = torch.arange(0,tensor.size(dim))
    
    return padded_tensor