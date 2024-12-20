import ipywidgets as widgets
from IPython.display import display, clear_output

input_features = {
    # "name", # The model can't handle freeform names
    "category",
    # "price",
    # "old_price",
    "sellable_online",
    "other_colors",
    # "short_description", # The model can't handle freeform descriptions
    "width",
    "height",
    "depth",
    "width_d",
    "height_d",
    "depth_d",
    "discounted",
    "discount_amount",
    "size"
}
checkboxes = [widgets.Checkbox(value=False, description=label) for label in input_features]
checkbox_container = widgets.VBox(children=checkboxes)
display(checkbox_container)

button = widgets.Button(description="Submit")

def on_button_clicked(b):
    selected_features = [box.description for box in checkboxes if box.value]
    print("Selected features:", selected_features)

button.on_click(on_button_clicked)
display(button)