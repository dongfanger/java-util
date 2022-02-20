import java.util.*;

/**
 * 100道算法题 Algorithm Question
 */
public class AQ80 {
    //======================简单题开始======================
    //1大数加法
    public String t1_bigNumberAdd(String s, String t) {
        // write code here
        Stack<Integer> stack = new Stack<>();
        StringBuilder result = new StringBuilder();
        int i = s.length() - 1;
        int j = t.length() - 1;
        int carry = 0;
        while (i >= 0 || j >= 0 || carry != 0) {
            carry += i >= 0 ? s.charAt(i--) - '0' : 0;
            carry += j >= 0 ? t.charAt(j--) - '0' : 0;
            stack.push(carry % 10);
            carry /= 10;
        }
        while (!stack.isEmpty()) {
            result.append(stack.pop());
        }
        return result.toString();
    }

    //2链表中环的入口结点
    public static class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ListNode t2_entryNodeOfLoop(ListNode pHead) {
        if (pHead == null) return null;
        ListNode slow = pHead;
        ListNode fast = pHead;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                break;
            }
        }

        if (fast == null || fast.next == null) {
            return null;
        }

        fast = pHead;

        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }

        return fast;

    }

    //3判断链表中是否有环
    public boolean t3_hasCycle(ListNode head) {
        ListNode pos = head;
        Set<ListNode> visited = new HashSet<>();
        while (pos != null) {
            if (visited.contains(pos)) {
                return true;
            } else {
                visited.add(pos);
            }
            pos = pos.next;
        }
        return false;
    }

    //4二叉树中的最大路径和
    public static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int i) {
            val = i;
        }
    }

    int maxSum = Integer.MIN_VALUE;

    public int t4_maxPathSum(TreeNode root) {
        // write code here
        maxGain(root);
        return maxSum;
    }

    public int maxGain(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftGain = Math.max(maxGain(node.left), 0);
        int rightGain = Math.max(maxGain(node.right), 0);

        int priceNewPath = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, priceNewPath);

        return node.val + Math.max(leftGain, rightGain);
    }

    //5将升序数组转化为平衡二叉搜索树
    public TreeNode t5_sortedArrayToBST(int[] num) {
        // write code here
        if (num == null || num.length == 0) {
            return null;
        }

        return process(num, 0, num.length - 1);
    }

    public TreeNode process(int[] num, int left, int right) {
        if (left > right) {
            return null;
        }

        if (left == right) {
            return new TreeNode(num[left]);
        }

        int len = right - left + 1;
        int mid = left + len / 2;
        TreeNode root = new TreeNode(num[mid]);
        root.left = process(num, left, mid - 1);
        root.right = process(num, mid + 1, right);
        return root;
    }

    //6重建二叉树
    public TreeNode t6_reConstructBinaryTree(int[] pre, int[] vin) {
        return reConstructBinaryTree(pre, 0, pre.length - 1, vin, 0, vin.length - 1);
    }

    private TreeNode reConstructBinaryTree(int[] pre, int startPre, int endPre, int[] vin, int startVin, int endVin) {
        if (startPre > endPre || startVin > endVin) {
            return null;
        }

        TreeNode root = new TreeNode(pre[startPre]);

        for (int i = startVin; i <= endVin; i++) {
            if (vin[i] == pre[startPre]) {
                root.left = reConstructBinaryTree(pre, startPre + 1, startPre + i - startVin, vin, startVin, i - 1);
                root.right = reConstructBinaryTree(pre, i - startVin + startPre + 1, endPre, vin, i + 1, endVin);
                break;
            }
        }

        return root;
    }

    //7按之字形顺序打印二叉树
    public ArrayList<ArrayList<Integer>> t7_print(TreeNode pRoot) {
        LinkedList<TreeNode> deque = new LinkedList<>();
        ArrayList res = new ArrayList<>();
        if (pRoot != null) {
            deque.add(pRoot);
        }
        while (!deque.isEmpty()) {
            ArrayList tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                TreeNode node = deque.removeFirst();
                tmp.add(node.val);
                if (node.left != null) {
                    deque.addLast(node.left);
                }
                if (node.right != null) {
                    deque.addLast(node.right);
                }
            }
            res.add(tmp);
            if (deque.isEmpty()) {
                break;
            }
            tmp = new ArrayList<>();
            for (int i = deque.size(); i > 0; i--) {
                TreeNode node = deque.removeLast();
                tmp.add(node.val);
                if (node.right != null) {
                    deque.addFirst(node.right);
                }
                if (node.left != null) {
                    deque.addFirst(node.left);
                }
            }
            res.add(tmp);
        }

        return res;
    }

    //8求二叉树的层序遍历
    public ArrayList<ArrayList<Integer>> t8_levelOrder(TreeNode root) {
        // write code here
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }

        dfs(root, res, 0);
        return res;
    }

    public void dfs(TreeNode root, ArrayList<ArrayList<Integer>> list, int depth) {
        if (root == null) {
            return;
        }

        if (list.size() == depth) {
            list.add(new ArrayList<>());
        }

        dfs(root.left, list, depth + 1);
        list.get(depth).add(root.val);
        dfs(root.right, list, depth + 1);
    }

    //9对称的二叉树
    boolean t9_isSymmetrical(TreeNode pRoot) {
        return isSame(pRoot, pRoot);
    }

    public boolean isSame(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        return root1.val == root2.val &&
                isSame(root1.left, root2.right) &&
                isSame(root1.right, root2.left);
    }

    //10最长回文子串
    public int t10_getLongestPalindrome(String A) {
        // write code here
        int maxLen = 0;
        int n = A.length();

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                String now = A.substring(i, j);
                if (isHuiWen(now) && now.length() > maxLen) {
                    maxLen = now.length();
                }
            }
        }

        return maxLen;
    }

    public boolean isHuiWen(String s) {
        int len = s.length();
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(i) != s.charAt(len - i - 1)) {
                return false;
            }
        }
        return true;
    }

    //11顺时针旋转矩阵
    public int[][] t11_rotateMatrix(int[][] mat, int n) {
        // write code here
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int tmp = mat[i][j];
                mat[i][j] = mat[j][i];
                mat[j][i] = tmp;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int tmp = mat[i][n - 1 - j];
                mat[i][n - 1 - j] = mat[i][j];
                mat[i][j] = tmp;
            }
        }

        return mat;
    }

    //12连续子数组的最大和
    public int t12_findGreatestSumOfSubArray(int[] array) {
        int res = array[0];
        int max = array[0];
        for (int i = 1; i < array.length; i++) {
            max = Math.max(max + array[i], array[i]);
            res = Math.max(max, res);
        }
        return res;
    }

    //13合并两个有序的数组
    public void t13_merge(int A[], int m, int B[], int n) {
        int i = 0, j = 0, p = 0;
        int[] c = new int[m + n];
        while (i < m && j < n) {
            c[p++] = A[i] <= B[j] ? A[i++] : B[j++];
        }
        while (i < m) {
            c[p++] = A[i++];
        }
        while (j < n) {
            c[p++] = B[j++];
        }
        for (int x = 0; x < p; x++) {
            A[x] = c[x];
        }
    }

    //14删除有序链表中重复的元素-I
    public ListNode t14_deleteDuplicates(ListNode head) {
        // write code here
        ListNode cur = head;
        while (cur != null) {
            while (cur.next != null && cur.val == cur.next.val) {
                cur.next = cur.next.next;
            }
            cur = cur.next;
        }
        return head;
    }

    //15删除有序链表中重复的元素-II
    public ListNode t15_deleteDuplicates(ListNode head) {
        // write code here
        ListNode node = new ListNode(-1);
        node.next = head;
        ListNode p = node;
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            if (cur.val != cur.next.val) {
                p = cur;
            } else {
                while (cur.next != null && cur.val == cur.next.val) {
                    cur = cur.next;

                }
                p.next = cur.next;

            }
            cur = cur.next;
        }
        return node.next;
    }

    //16括号生成
    public ArrayList<String> t16_generateParenthesis (int n) {
        // write code here
        ArrayList<String> result = new ArrayList<>(10);
        backtrack("", 0, 0, n, result);
        return result;
    }

    private void backtrack(String string, int open, int close, int n, List<String> result) {
        if (string.length() == n << 1) {
            result.add(string);
            return;
        }
        if (open < n) {
            backtrack(string + "(", open + 1, close, n ,result);
        }
        if (close < open) {
            backtrack(string + ")", open, close + 1, n, result);
        }
    }

    //17集合的所有子集(一)
    ArrayList<ArrayList<Integer>> result = new ArrayList<>();
    public ArrayList<ArrayList<Integer>> t17_subsets(int[] nums) {
        ArrayList<Integer> list = new ArrayList<>();
        Arrays.sort(nums);
        for (int j = 0; j <= nums.length; j++) {
            backtracking(nums, j, 0, list);
        }
        return result;
    }
    public void backtracking(int[] nums, int k, int start, ArrayList<Integer> list) {
        if (k < 0) {
            return;
        } else if (k == 0) {
            result.add(new ArrayList(list));
        } else {
            for (int i = start; i < nums.length; i++) {
                list.add(nums[i]);
                backtracking(nums, k - 1, i + 1, list);
                list.remove(list.size() - 1);
            }
        }
    }

    //18最小覆盖子串
    public String t18_minWindow (String S, String T) {
        // write code here
        int[] map = new int[128];
        for (int i = 0; i < T.length(); i++) {
            map[T.charAt(i)]++;
        }

        int begin = 0, end = 0, d = Integer.MAX_VALUE, counter = T.length(), head = 0;
        while (end < S.length()) {
            if (map[S.charAt(end++)]-- > 0) {
                counter--;
            }
            while(counter == 0) {
                if (end - begin < d) {
                    d = end - (head = begin);
                }
                if (map[S.charAt(begin++)]++ == 0) {
                    counter++;
                }
            }
        }
        return d == Integer.MAX_VALUE ? "" : S.substring(head, head + d);
    }

    //19缺失的第一个正整数
    public int t19_minNumberDisappeared (int[] nums) {
        // write code here
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0) {
                nums[i] = nums.length + 1;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            int x = Math.abs(nums[i]);
            if (x <= nums.length) {
                nums[x - 1] = (-1) * Math.abs(nums[x - 1]);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }

    //20第一个只出现一次的字符


    //冒泡排序
    public int[] bubbleSort(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = 0; j < nums.length - i - 1; j++) {
                if (nums[j] > nums[j + 1]) {
                    int temp = nums[j];
                    nums[j] = nums[j + 1];
                    nums[j + 1] = temp;
                }
            }
        }
        return nums;
    }


    //======================简单题结束======================

    //======================中等题开始======================
    //======================中等题结束======================
}
