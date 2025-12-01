#!/bin/bash

DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
API_ROOT="$(dirname "$DOCKER_DIR")"
API_ENV="$API_ROOT/.env"
DOCKER_ENV="$DOCKER_DIR/.env"

get_os() {
    unameOut="$(uname -s)"
    case "${unameOut}" in
        Linux*)     os=Linux;;
        Darwin*)    os=Mac;;
        CYGWIN*)    os=Cygwin;;
        MINGW*)     os=MinGw;;
        MSYS_NT*)   os=Git;;
        *)          os="UNKNOWN:${unameOut}"
    esac
    echo "${os}"
}

export OS
OS=$(get_os)

color_echo() {
    Color_Off="\033[0m"
    case "$1" in
        "green") echo -e "\033[1;92m$2$Color_Off";;
        "red") echo -e "\033[1;91m$2$Color_Off";;
        "blue") echo -e "\033[1;94m$2$Color_Off";;
        "yellow") echo -e "\033[1;93m$2$Color_Off";;
        "purple") echo -e "\033[1;95m$2$Color_Off";;
        "cyan") echo -e "\033[1;96m$2$Color_Off";;
        *) echo "$2";;
    esac
}

echo_title(){
    sep_line="========================================"
    len_title=${#1}

    if [ "$len_title" -gt 40 ]; then
        sep_line=$(printf "%0.s=" $(seq 1 $len_title))
        title="$1"
    else
        diff=$((38 - len_title))
        half_diff=$((diff / 2))
        sep=$(printf "%0.s=" $(seq 1 $half_diff))

        if [ $((diff % 2)) -ne 0 ]; then
            title="$sep $1 $sep="
        else
            title="$sep $1 $sep"
        fi
    fi

    color_echo purple "\n\n$sep_line\n$title\n$sep_line"
}

# the sed at the end removes trailing non-alphanumeric chars.
generate_random_string() {
    echo "$(openssl rand -base64 32 | tr -d '/\n' | sed -r -e "s/[^a-zA-Z0-9]+$//")"
}

prompt_user() {
    env_var=$(color_echo 'red' "$1")
    default_val="$2"
    current_val="$3"
    desc="$4"

    if [ "$default_val" != "$current_val" ]; then
        prompt="Press enter for $(color_echo 'cyan' "$default_val")"
    elif [ -n "$current_val" ]; then
        prompt="Press enter to keep $(color_echo 'cyan' "$current_val")"
    else
        prompt="Enter value"
    fi

    prompt="$prompt / type a space to set empty"
    read -p "$env_var $desc"$'\n'"$prompt: " value </dev/tty

    if [ "$value" = " " ]; then
        export new_value=""  # if user entered a space character, return empty value
    else
        export new_value="${value:-$default_val}"
    fi
}

sed_repl_inplace() {
    sed_expr="$1"
    file="$2"

    if [ "$OS" = "Linux" ]; then
        sed -i -e "$sed_expr" "$file"
    else
        sed -i "" -e "$sed_expr" "$file"
    fi
}

sudo_sed_repl_inplace() {
    sed_expr="$1"
    file="$2"

    if [ "$OS" = "Linux" ]; then
        [ -n "$SUDO_PSW" ] && echo "$SUDO_PSW" | sudo -S sed -i -e "$sed_expr" "$file" || sudo sed -i -e "$sed_expr" "$file"
    else
        [ -n "$SUDO_PSW" ] && echo "$SUDO_PSW" | sudo -S sed -i "" -e "$sed_expr" "$file" || sudo sed -i "" -e "$sed_expr" "$file"
    fi
}

DEFAULT_PARAMS=()
is_in_default_params() {
    local param=$1
    for default_param in "${DEFAULT_PARAMS[@]}"; do
        if [ "$param" = "$default_param" ]; then
            return 0
        fi
    done
    return 1
}

get_template_hash() {
    local template_file=$1
    md5sum "$template_file" | awk '{print $1}'
}

store_template_hash() {
    local template_file=$1
    local hash_file="${template_file}.hash"
    local current_hash=$(get_template_hash "$template_file")
    echo "$current_hash" > "$hash_file"
}

check_template_hash() {
    local template_file=$1
    local hash_file="${template_file}.hash"

    if [ ! -f "$hash_file" ]; then
        store_template_hash "$template_file"
        return 1  # Hash file didn't exist, template is new
    fi

    local stored_hash=$(cat "$hash_file")
    local current_hash=$(get_template_hash "$template_file")

    if [ "$stored_hash" != "$current_hash" ]; then
        store_template_hash "$template_file"
        return 1  # Hash changed
    fi

    return 0  # Hash unchanged
}

get_env_value() {
    param=$1
    env_file=$2
    value=$(awk -F= -v param="$param" '/^[^#]/ && $1 == param {gsub(/"/, "", $2); print $2}' "$env_file")
    echo "$value"
}

get_env_desc() {
    current_line="$1"
    prev_line="$2"
    desc=""
    if [[ $prev_line =~ ^# ]]; then
        desc=$(echo "$prev_line" | sed 's/^#\s*//')
    fi
    echo "$desc"
}

get_default_val() {
    local param=$1
    if [[ "$param" = "PROD_API_URL" ]]; then
        default_val=${PROD_API_URL:-""}
    elif [ -n "${!param}" ]; then
        # if the value is already exported in the current shell, use it as default
        default_val="${!param}"
    elif [[ "$param" =~ ^.*(PASSWORD|SECRET).*$ ]]; then
        default_val="$(generate_random_string)"
    elif [[ "$param" = "DOCKER" ]]; then
        if [[ "$(get_env_value "TARGET" "$API_ENV")" = "prod" ]]; then
            default_val="True"
        else
            default_val="False"
        fi
    elif [[ "$param" = "API_DATA_FOLDER" ]]; then
        if [[ "$(get_env_value "DOCKER" "$API_ENV")" = "True" ]]; then
            default_val="/data/"
        else
            default_val="data/"
        fi
    elif [[ "$param" = "YOLO_CONFIG_DIR" ]]; then
        if [[ "$(get_env_value "DOCKER" "$API_ENV")" = "True" ]]; then
            default_val="/data/yolotmp/"
        else
            default_val="data/yolotmp/"
        fi
    else
        default_val=$(get_env_value "$param" "$env_file")
    fi
    echo "$default_val"
}

update_env_var() {
    local value=$1
    local param=$2
    local env_file=$3
    sed_repl_inplace "s~^$param=.*~$param=$value~" "$env_file"
}

update_env() {
    local env_file=$1

    local prev_line=""
    while IFS= read -r line; do
        if [[ $line =~ ^[^#]*= ]]; then
            param=$(echo "$line" | cut -d'=' -f1)
            desc=$(get_env_desc "$line" "$prev_line")
            default_val=$(get_default_val $param)
            current_val=$(get_env_value "$param" "$env_file")

            if [ "$INSTALL_MODE" = "full_install" ]; then
                # For full install, all variables are prompted
                prompt_user "$param" "$default_val" "$current_val" "$desc"
            elif [ -n "${!param}" ]; then
                # If variable is already set in the current shell, use it as default
                new_value="${!param}"
            elif is_in_default_params "$param"; then
                # If param is in default params, use default value if it exists
                new_value="$default_val"
            else
                prompt_user "$param" "$default_val" "$current_val" "$desc"
            fi

            update_env_var "$new_value" "$param" "$env_file"
        fi
        prev_line="$line"
    done < "$env_file"
}

export_env() {
    set -a # Turn on allexport mode
    source "$env_file"
    set +a # Turn off allexport mode
}

setup_env() {
    local env_file=$1
    local template_file="${env_file}.template"
    local default_params=("${@:2}")  # All arguments after $1 are default params
    DEFAULT_PARAMS=("${default_params[@]}")

    if [ ! -f "$env_file" ]; then
        color_echo yellow "\nCreating $env_file"
        cp "$template_file" "$env_file"
    elif ! check_template_hash "$template_file"; then
        color_echo yellow "\nUpdating $env_file"
        # the env file has already been created, but the template has changed
        export_env "$env_file" # source current values to copy them in new env
        cp "$env_file" "${env_file}.backup"
        cp "$template_file" "$env_file"
    else
        options=("yes" "no")

        color_echo yellow "\n$env_file is up-to-date. Do you want to regenerate it again?"
        answer=$(printf "%s\n" "${options[@]}" | fzy)
        if [ "$answer" = "yes" ]; then
            rm "${template_file}.hash"
            setup_env $env_file
            exit 0
        fi
        color_echo yellow "\nSkipping $env_file update..."
        export_env "$env_file" "${DEFAULT_PARAMS[@]}"
        exit 0
    fi

    if [ -z "$INSTALL_MODE" ]; then
        select_install_mode
    fi

    update_env "$env_file"
    export_env "$env_file"
}
