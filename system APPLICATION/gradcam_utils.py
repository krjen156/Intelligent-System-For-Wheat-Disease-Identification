import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm


def find_last_conv_layer_in_layers(layers):
    for layer in reversed(layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def find_nested_base_model(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            return layer, i
    return None, None


def safe_call_layer(layer, x):
    if isinstance(x, (list, tuple)):
        x = x[0]

    # Check what input dimension the layer expects
    expected_ndim = None
    input_spec = getattr(layer, "input_spec", None)

    if input_spec is not None:
        if isinstance(input_spec, (list, tuple)):
            input_spec = input_spec[0]
        expected_ndim = getattr(input_spec, "ndim", None)

    # Only flatten/pool when the next layer expects 2D input
    if expected_ndim == 2 and len(x.shape) > 2:
        x = tf.reduce_mean(x, axis=list(range(1, len(x.shape) - 1)))

    # Fallback: Dense layers always need 2D input
    elif isinstance(layer, tf.keras.layers.Dense) and len(x.shape) > 2:
        x = tf.reduce_mean(x, axis=list(range(1, len(x.shape) - 1)))

    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def normalize_heatmap(heatmap):
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        return None

    heatmap = heatmap / max_value
    return heatmap.numpy()


def make_gradcam_heatmap(img_array, model, pred_index):
    """
    Works for:
    - nested transfer-learning models: input -> VGG16/ResNet50 -> head
    - sequential custom CNNs: input -> Conv2D -> ... -> Dense
    """

    # Build model once if needed
    _ = model(img_array, training=False)

    nested_base, nested_index = find_nested_base_model(model)

    # Case 1: nested base model, e.g. VGG16 or ResNet50
    if nested_base is not None:
        last_conv = find_last_conv_layer_in_layers(nested_base.layers)

        if last_conv is None:
            return None

        conv_model = tf.keras.Model(
            inputs=nested_base.input,
            outputs=[last_conv.output, nested_base.output]
        )

        classifier_layers = model.layers[nested_index + 1:]

        with tf.GradientTape() as tape:
            conv_outputs, x = conv_model(img_array)

            for layer in classifier_layers:
                x = safe_call_layer(layer, x)

            predictions = x

            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            if len(predictions.shape) != 2:
                return None

            if pred_index >= predictions.shape[1]:
                return None

            class_output = predictions[:, pred_index]

        grads = tape.gradient(class_output, conv_outputs)

    # Case 2: Sequential/custom CNN, e.g. WheatNetwork
    else:
        last_conv = find_last_conv_layer_in_layers(model.layers)

        if last_conv is None:
            return None

        last_conv_index = model.layers.index(last_conv)

        try:
            conv_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=last_conv.output
            )
        except Exception:
            return None

        classifier_layers = model.layers[last_conv_index + 1:]

        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_array)
            x = conv_outputs

            for layer in classifier_layers:
                x = safe_call_layer(layer, x)

            predictions = x

            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            if len(predictions.shape) != 2:
                return None

            if pred_index >= predictions.shape[1]:
                return None

            class_output = predictions[:, pred_index]

        grads = tape.gradient(class_output, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    return normalize_heatmap(heatmap)


def overlay_gradcam(original_image, heatmap, input_size, alpha=0.45):
    original = np.array(original_image.convert("RGB").resize(input_size))

    heatmap = cv2.resize(heatmap, input_size)
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = np.uint8(255 * jet_heatmap)

    overlay = cv2.addWeighted(original, 1 - alpha, jet_heatmap, alpha, 0)
    return overlay