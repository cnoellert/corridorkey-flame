PYBOX_DIR  = /opt/corridorkey/pybox
REF_DIR    = /opt/corridorkey/reference
COMFYUI_CK = /Users/cnoellert/Documents/GitHub/comfyui-corridorkey

PYBOX_FILES = corridorkey_pybox.py \
              corridorkey_daemon_mlx.py \
              corridorkey_daemon_cuda.py

.PHONY: install install-ref

install:
	mkdir -p $(PYBOX_DIR)
	cp $(PYBOX_FILES) $(PYBOX_DIR)/
	@echo "Installed pybox to $(PYBOX_DIR)"

# Copy PyTorch reference inference code from comfyui-corridorkey repo.
# Run once on Mac; on Rocky Linux, install manually from the same repo.
install-ref:
	mkdir -p $(REF_DIR)
	cp -r $(COMFYUI_CK)/reference/CorridorKeyModule $(REF_DIR)/
	cp -r $(COMFYUI_CK)/utils $(REF_DIR)/
	@echo "Installed reference inference code to $(REF_DIR)"
